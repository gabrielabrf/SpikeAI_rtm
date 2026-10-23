"""RTMPose HALPE26 -> detector de contato oficial do Footskate Reducer."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from urllib.request import urlopen

import numpy as np

COMMIT = "1bf72e52838a2629e59128d6a49025077303b799"
BASE_URL = f"https://raw.githubusercontent.com/vt-vl-lab/footskate_reducer/{COMMIT}/ground_detector"
CHECKPOINT_SHA256 = "a51cd35847d9d8d2aff2cae1a295c0fac2945e39102b0eefde93f2cc90e44427"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "footskate"
CONTACT_NAMES = ("left_toe", "left_heel", "right_toe", "right_heel")
CONTACT_POINTS = (19, 21, 22, 24)  # BODY25, na ordem da saída da rede.
MIN_CONF = 0.3

# Índice BODY25 -> índice HALPE26. Calcanhares e dedos são pontos reais.
BODY25_FROM_HALPE26 = [
    0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11,
    13, 15, 2, 1, 4, 3, 20, 22, 24, 21, 23, 25,
]


def to_body25(xy, scores):
    if xy.shape != (26, 2) or scores.shape != (26,):
        raise ValueError("São necessários os 26 pontos HALPE26, incluindo os pés.")
    pose = np.column_stack((xy, scores))[BODY25_FROM_HALPE26].astype(np.float32)
    invalid = ~np.isfinite(pose).all(axis=1) | (pose[:, 2] < 0)
    pose[invalid] = 0
    return pose


def download_models():
    """Baixa só os arquivos necessários, de uma versão fixa do repositório."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for remote, name in [
        ("net.py", "net.py"),
        ("op2npy.py", "op2npy.py"),
        ("pretrained/ckpt/model_best.pth", "model_best.pth"),
    ]:
        path = MODEL_DIR / name
        if not path.exists():
            print(f"Baixando Footskate Reducer: {name}", flush=True)
            temporary = path.with_suffix(".download")
            try:
                with urlopen(f"{BASE_URL}/{remote}", timeout=120) as response:
                    with temporary.open("wb") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    checkpoint = MODEL_DIR / "model_best.pth"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(f"Peso do Footskate inválido. Apague {checkpoint} e execute novamente.")


def smooth_poses(poses):
    """Interpolação e OneEuroFilter originais; confiança observada é preservada."""
    tree = ast.parse((MODEL_DIR / "op2npy.py").read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"LowPassFilter", "OneEuroFilter"}]
    namespace = {"np": np}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "op2npy.py", "exec"), namespace)
    result = poses.copy()
    for joint in range(25):
        valid = np.flatnonzero(poses[:, joint, 2] >= MIN_CONF)
        for axis in range(2):
            values = (np.interp(np.arange(len(poses)), valid, poses[valid, joint, axis])
                      if len(valid) else np.zeros(len(poses)))
            euro = namespace["OneEuroFilter"](freq=120, mincutoff=1.0, beta=0.3, dcutoff=1.0)
            result[:, joint, axis] = [euro(float(value)) for value in values]
    return result


def temporal_inputs(poses):
    """Mesma janela de 7 frames e normalização do DatasetInference oficial."""
    inputs = np.zeros((len(poses), 7, 16, 2), dtype=np.float32)
    valid = np.zeros(len(poses), dtype=bool)
    for frame in range(len(poses)):
        indices = np.clip(np.arange(frame - 3, frame + 4), 0, len(poses) - 1)
        window = poses[indices].copy()
        window[:, :, :2] -= poses[frame, 8, :2]
        window = window[:, 9:25]
        selected = window[3, :, 2] > 0.2
        if selected.sum() < 2:
            continue
        extent = np.linalg.norm(np.ptp(window[3, selected, :2], axis=0))
        if extent < 1:
            continue
        inputs[frame] = window[:, :, :2] * (150.0 / extent)
        valid[frame] = True
    return inputs, valid


class FootskateDetector:
    def __init__(self):
        import torch

        download_models()
        spec = importlib.util.spec_from_file_location("footskate_net", MODEL_DIR / "net.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = module.BasicTemporalModel(in_channels=32, time_window=3)
        checkpoint = torch.load(MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def predict(self, poses, threshold=0.5):
        import torch

        inputs, valid = temporal_inputs(smooth_poses(poses))
        probabilities = np.zeros((len(poses), 4), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(poses), 128):
                batch = torch.from_numpy(inputs[start:start + 128])
                probabilities[start:start + len(batch)] = self.model(batch).numpy()
        return probabilities, contact_states(probabilities, poses, valid, threshold)


def contact_states(probabilities, poses, valid, threshold=0.5):
    """1: apoio observado; 0: ambos os pés livres; -1: evidência insuficiente."""
    confidence = poses[:, :, 2]
    lower_body = (confidence[:, [8, 9, 10, 11, 12, 13, 14]] >= MIN_CONF).all(axis=1)
    coverage = (confidence[:, 9:25] >= MIN_CONF).mean(axis=1) >= 0.65
    # Um frame sem atleta na janela temporal também invalida sua vizinhança.
    reliable = valid & lower_body & coverage
    context_ok = np.array([
        reliable[max(0, i - 3):min(len(poses), i + 4)].all()
        for i in range(len(poses))
    ])
    regions = np.column_stack([
        np.minimum(confidence[:, 19], confidence[:, 20]), confidence[:, 21],
        np.minimum(confidence[:, 22], confidence[:, 23]), confidence[:, 24],
    ]) >= MIN_CONF
    regions &= context_ok[:, None]
    positive = probabilities >= threshold
    states = np.full(len(poses), -1, dtype=np.int8)
    states[(regions & positive).any(axis=1)] = 1
    states[regions.all(axis=1) & ~positive.any(axis=1)] = 0
    return states
