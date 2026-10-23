"""Um clipe, um atacante, dois eventos: take_off e landing."""

import argparse
import os
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Clipe com um ataque completo.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selecionar", action="store_true", help="Marcar o atacante na imagem inicial.")
    selection.add_argument("--atleta", type=int, help="Índice inicial mostrado na imagem de seleção.")
    parser.add_argument("--saida", type=Path, help="Pasta de saída. Padrão: output/NOME_DO_VIDEO.")
    parser.add_argument("--limiar", type=float, default=0.5, help="Limiar de contato do Footskate (padrão: 0.5).")
    return parser.parse_args()


def load_pose_model():
    """RTMPose-m com os pontos dos pés (HALPE26), limitado a poucas threads."""
    import cv2
    import onnxruntime as ort
    from rtmlib import BodyWithFeet

    model = BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu", to_openpose=False)
    # Evita excesso de threads em CPUs pequenas.
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    for tool in (model.det_model, model.pose_model):
        tool.session = ort.InferenceSession(tool.onnx_model, options, providers=["CPUExecutionProvider"])
    cv2.setNumThreads(1)
    return model


def extract_poses(video, pose_model, out, athlete=None, interactive=False):
    """Pose BODY25 do atacante e tempo (s) de cada frame, com os frames em ordem."""
    import av
    import numpy as np

    from attacker import AttackerTracker, choose_attacker
    from footskate import to_body25

    poses, timestamps = [], []
    tracker = None
    athlete_id = None
    first_time = None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        total = stream.frames
        for i, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError("O vídeo não fornece os tempos dos frames (PTS).")
            time_s = float(frame.pts * frame.time_base)
            if first_time is None:
                first_time = time_s
            timestamp = time_s - first_time
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("O vídeo tem tempos de frames repetidos ou fora de ordem.")
            image = frame.to_ndarray(format="bgr24")
            boxes = np.asarray(pose_model.det_model(image), dtype=float).reshape(-1, 4)
            if i == 0:
                athlete_id = choose_attacker(image, boxes, athlete, interactive, out / "athletes.jpg")
                tracker = AttackerTracker(boxes[athlete_id], max_missing=max(2, round(0.3 * (fps or 30))))
                selected = athlete_id
                print(f"Atacante inicial: {athlete_id}. Calculando os pontos...", flush=True)
            else:
                selected = tracker.update(boxes)

            pose = np.zeros((25, 3), dtype=np.float32)
            if selected is not None:
                xy, scores = pose_model.pose_model(image, bboxes=[boxes[selected]])
                pose = to_body25(xy[0], scores[0])
                # Pontos projetados fora da imagem não são observações válidas.
                outside = ((pose[:, 0] < 0) | (pose[:, 0] >= image.shape[1]) |
                           (pose[:, 1] < 0) | (pose[:, 1] >= image.shape[0]))
                pose[outside, 2] = 0
            poses.append(pose)
            timestamps.append(timestamp)
            if i % 30 == 0:
                print(f"Frame {i}" + (f"/{total}" if total else ""), flush=True)

    if not poses:
        raise ValueError("O vídeo não contém frames decodificáveis.")
    return np.asarray(poses), np.asarray(timestamps), athlete_id, fps


def run(args):
    from algorithm_rules import detect_events
    from export_handler import save_data, save_video
    from footskate import FootskateDetector
    from ground import refine_with_ground
    from validator import validate_video

    check = validate_video(args.video)
    if check["status"] == "REJEITAR":
        raise ValueError(" ".join(check["motivos_rejeicao"]))
    if not 0 < args.limiar < 1:
        raise ValueError("O limiar deve estar entre 0 e 1.")
    if args.selecionar and sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        raise ValueError("Sem interface gráfica. Execute sem --selecionar e use a imagem gerada para escolher --atleta N.")
    out = args.saida or Path("output") / args.video.stem
    if any((out / name).exists() for name in ("events.json", "inspection.mp4")):
        raise ValueError(f"{out} já tem um resultado. Escolha outra pasta com --saida.")
    for alert in check["alertas"]:
        print(f"Aviso: {alert}")

    print("Carregando Footskate Reducer e RTMPose com pontos dos pés...", flush=True)
    contact_model = FootskateDetector()  # Falha antes de processar o vídeo se faltar torch ou o peso.
    pose_model = load_pose_model()
    out.mkdir(parents=True, exist_ok=True)

    poses, timestamps, athlete_id, fps = extract_poses(args.video, pose_model, out, args.atleta, args.selecionar)
    print("Calculando contato com o Footskate Reducer...", flush=True)
    probabilities, model_states = contact_model.predict(poses, args.limiar)
    print("Conferindo o contato com a referência local do piso...", flush=True)
    evidence_states, ground_lines, ground_info, foot_states = refine_with_ground(
        args.video, poses, probabilities, args.limiar)
    hybrid = evidence_states is not None
    if not hybrid:
        print(f"Referência do piso indisponível: {ground_info['reason']} Usando Footskate.")
        evidence_states = model_states
    events, states = detect_events(evidence_states, timestamps,
                                   source="footskate+ground_reference" if hybrid else "footskate")
    print("Gerando vídeo de inspeção...", flush=True)
    save_video(args.video, out / "inspection.mp4", poses, probabilities, states, timestamps, events,
               args.limiar, ground_lines, foot_states, hybrid)
    data = save_data(args.video, out, athlete_id, fps, timestamps, probabilities, states, evidence_states,
                     events, args.limiar, ground_info, check["alertas"])
    for event in events:
        if event["frame"] is None:
            print(f"{event['event']}: INCONCLUSIVO. {event['reason']}")
        else:
            print(f"{event['event']}: {event['status']} | frame {event['frame']} | {event['time_s']:.3f} s | intervalo {event['frame_range']}")
    print(f"Status: {data['pipeline_status']}")
    print(f"Resultado: {out.resolve()}")


if __name__ == "__main__":
    args = arguments()
    try:
        run(args)
    except ModuleNotFoundError as error:
        raise SystemExit(f"Dependência ausente: {error.name}. Execute bash instalar.sh e ative a .venv.") from None
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Erro: {error}") from None
