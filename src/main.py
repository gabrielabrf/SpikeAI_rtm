"""Um clipe, um atacante, quatro eventos do ataque de vôlei."""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Tuple

# Garante que o Python encontre os ficheiros da pasta src
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Clipe com um ataque completo ou deixe vazio para processar a pasta input/.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--selecionar",
        action="store_true",
        help="Marcar o atacante na imagem inicial.",
    )
    selection.add_argument(
        "--atleta",
        type=int,
        help="Índice inicial mostrado na imagem de seleção.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        help="Pasta de saída. Padrão: output/NOME_DO_VIDEO.",
    )
    parser.add_argument(
        "--limiar",
        type=float,
        default=0.5,
        help="Limiar de contato do Footskate (padrão: 0.5).",
    )
    return parser.parse_args()


def load_pose_model():
    """RTMPose-m com os pontos dos pés (HALPE26), limitado a poucas threads."""
    import cv2
    import onnxruntime as ort
    from rtmlib import BodyWithFeet

    model = BodyWithFeet(
        mode="balanced", backend="onnxruntime", device="cpu", to_openpose=False
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1

    for tool in (model.det_model, model.pose_model):
        tool.session = ort.InferenceSession(
            tool.onnx_model, options, providers=["CPUExecutionProvider"]
        )
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
                athlete_id = choose_attacker(
                    image, boxes, athlete, interactive, out / "athletes.jpg"
                )
                tracker = AttackerTracker(
                    boxes[athlete_id], max_missing=max(2, round(0.3 * (fps or 30)))
                )
                selected = athlete_id
                print(f"Atacante inicial: {athlete_id}. Calculando os pontos...", flush=True)
            else:
                selected = tracker.update(boxes)

            pose = np.zeros((25, 3), dtype=np.float32)
            if selected is not None:
                xy, scores = pose_model.pose_model(image, bboxes=[boxes[selected]])
                pose = to_body25(xy[0], scores[0])
                outside = (
                    (pose[:, 0] < 0)
                    | (pose[:, 0] >= image.shape[1])
                    | (pose[:, 1] < 0)
                    | (pose[:, 1] >= image.shape[0])
                )
                pose[outside, 2] = 0
                if not tracker.confirm_pose(pose):
                    pose.fill(0)

            poses.append(pose)
            timestamps.append(timestamp)

            if i % 30 == 0:
                print(
                    f"Frame {i}" + (f"/{total}" if total else ""),
                    flush=True,
                )

    if not poses:
        raise ValueError("O vídeo não contém frames decodificáveis.")

    return np.asarray(poses), np.asarray(timestamps), athlete_id, fps


class SpikeArmAnalyzer:
    """Quatro eventos do ataque: saída, preparação máxima, braço à frente e pouso.

    take_off e landing vêm do contato dos pés e permanecem eventos próprios.
    A mão mais recuada e o início do avanço são medidos só entre esses dois.
    """

    def __init__(
        self,
        video_path: Path,
        events_base: List[Dict[str, Any]],
        poses: np.ndarray,
        timestamps: np.ndarray,
        fps: float,
    ):
        self.video_path = video_path
        self.events_base = events_base
        self.poses = poses
        self.timestamps = timestamps
        self.fps = fps if fps > 0 else 30.0

    def _foot_event(self, name: str) -> Dict[str, Any]:
        for event in self.events_base:
            if event.get("event") == name:
                return event
        return {
            "event": name,
            "status": "inconclusive",
            "frame": None,
            "time_s": None,
            "frame_range": None,
            "time_range_s": None,
            "source": None,
            "reason": "Não detectado.",
        }

    def _anchor(self, event: Dict[str, Any], edge: int) -> Optional[int]:
        if event.get("frame") is not None:
            return int(event["frame"])
        bounds = event.get("frame_range")
        if bounds and bounds[0] is not None and bounds[1] is not None:
            return int(bounds[edge])
        return None

    def _wrist_series(self, shoulder: int, wrist: int, start: int, end: int):
        frames, relative = [], []
        for index in range(start, end + 1):
            pose = self.poses[index]
            if pose[shoulder, 2] > 0.2 and pose[wrist, 2] > 0.2:
                frames.append(index)
                relative.append(pose[wrist, :2] - pose[shoulder, :2])
        if len(frames) < 5:
            return None
        relative = np.asarray(relative, dtype=float)
        centered = relative - relative.mean(axis=0)
        if np.linalg.norm(centered) < 1:
            return None
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        projection = relative @ axis
        padded = np.pad(projection, 1, mode="edge")
        smooth = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
        speed = np.zeros(len(frames))
        for index in range(1, len(frames)):
            dt = float(self.timestamps[frames[index]] - self.timestamps[frames[index - 1]])
            if dt > 0:
                speed[index] = (smooth[index] - smooth[index - 1]) / dt
        if np.max(speed) < -np.min(speed):
            smooth = -smooth
            speed = -speed
        return frames, smooth, speed

    def _detect_arm(self, take_off: Dict[str, Any], landing: Dict[str, Any]):
        start = self._anchor(take_off, 0)
        end = self._anchor(landing, 1)
        if start is None or end is None or end - start < 4:
            return None, None
        start = max(0, start)
        end = min(len(self.poses) - 1, end)
        candidates = []
        for shoulder, wrist in ((2, 4), (5, 7)):
            series = self._wrist_series(shoulder, wrist, start, end)
            if series is None:
                continue
            frames, projection, speed = series
            peak = int(np.argmax(speed))
            if peak < 1 or speed[peak] < 80 or projection.max() - projection.min() < 10:
                continue
            prep = int(np.argmin(projection[: peak + 1]))
            forward = next(
                (index for index in range(prep + 1, len(speed)) if speed[index] >= 0.15 * speed[peak]),
                None,
            )
            if forward is None:
                continue
            candidates.append((float(speed[peak]), frames[prep], frames[forward]))
        if not candidates:
            return None, None
        _, prep_frame, forward_frame = max(candidates)
        return prep_frame, forward_frame

    def _arm_event(self, name: str, frame: Optional[int], reason: str) -> Dict[str, Any]:
        found = frame is not None
        return {
            "event": name,
            "status": "detected" if found else "inconclusive",
            "frame": frame,
            "time_s": round(float(self.timestamps[frame]), 6) if found else None,
            "frame_range": None,
            "time_range_s": None,
            "source": "attacking arm",
            "reason": None if found else reason,
        }

    def process_arm_events(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        take_off = self._foot_event("take_off")
        landing = self._foot_event("landing")
        prep_frame, forward_frame = self._detect_arm(take_off, landing)
        missing = "Saída ou pouso sem intervalo suficiente para localizar o braço."
        if self._anchor(take_off, 0) is not None and self._anchor(landing, 1) is not None:
            missing = "Mão do braço de ataque não pôde ser localizada entre a saída e o pouso."
        final_events = [
            take_off,
            self._arm_event(
                "preparacao_maxima_aparente",
                prep_frame,
                missing,
            ),
            self._arm_event(
                "braco_frente_inicio",
                forward_frame,
                missing,
            ),
            landing,
        ]
        return final_events, {event["event"]: event for event in final_events}


def run(args):
    from algorithm_rules import detect_events
    from export_handler import save_data, save_video
    from footskate import FootskateDetector
    from ground import refine_with_ground
    from validator import validate_video

    check = validate_video(args.video)
    if check["status"] == "REJEITAR":
        raise ValueError(" ".join(check["motivos_rejeicao"]))

    if not (0 < args.limiar < 1):
        raise ValueError("O limiar deve estar entre 0 e 1.")

    if (
        args.selecionar
        and sys.platform.startswith("linux")
        and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    ):
        raise ValueError(
            "Sem interface gráfica. Execute sem --selecionar e use a imagem gerada para escolher --atleta N."
        )

    out = args.saida or (PROJECT_ROOT / "output" / args.video.stem)
    if any((out / name).exists() for name in ("events.json", "inspection.mp4")):
        print(f"Aviso: {out} já tem um resultado. Atualizando ficheiros...")

    for alert in check.get("alertas", []):
        print(f"Aviso: {alert}")

    print("Carregando Footskate Reducer e RTMPose com pontos dos pés...", flush=True)
    contact_model = FootskateDetector()
    pose_model = load_pose_model()

    out.mkdir(parents=True, exist_ok=True)
    poses, timestamps, athlete_id, fps = extract_poses(
        args.video, pose_model, out, args.atleta, args.selecionar
    )

    print("Calculando contato com o Footskate Reducer...", flush=True)
    probabilities, model_states = contact_model.predict(poses, args.limiar)

    print("Conferindo o contato com a referência local do piso...", flush=True)
    evidence_states, ground_lines, ground_info, foot_states = refine_with_ground(
        args.video, poses, probabilities, args.limiar
    )

    hybrid = evidence_states is not None
    if not hybrid:
        print(
            f"Referência do piso indisponível: {ground_info.get('reason', '')}. Usando Footskate."
        )
        evidence_states = model_states

    events_base, states = detect_events(
        evidence_states,
        timestamps,
        source="footskate+ground reference" if hybrid else "footskate",
    )

    arm_analyzer = SpikeArmAnalyzer(
        args.video, events_base, poses, timestamps, fps
    )
    events, events_summary = arm_analyzer.process_arm_events()

    print("Gerando vídeo de inspeção...", flush=True)
    save_video(
        args.video,
        out / "inspection.mp4",
        poses,
        probabilities,
        states,
        timestamps,
        events,
        args.limiar,
        ground_lines,
        foot_states,
        hybrid,
    )

    data = save_data(
        args.video,
        out,
        athlete_id,
        fps,
        timestamps,
        probabilities,
        states,
        evidence_states,
        events,
        args.limiar,
        ground_info,
        check.get("alertas", []),
    )
    data["events_4_summary"] = events_summary

    with open(out / "events.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n--- RESUMO DOS 4 EVENTOS DO ATAQUE ---")
    for event in events:
        if event.get("frame") is None or event.get("status") == "INCONCLUSIVO":
            print(f". {event['event']}: INCONCLUSIVO. {event.get('reason', '')}")
        else:
            time_val = (
                event["time_s"] if event.get("time_s") is not None else 0.0
            )
            sub_info = (
                f" [{event['sub_type'].upper()}]" if event.get("sub_type") else ""
            )
            print(
                f". {event['event']}{sub_info}: {event['status']} | frame {event['frame']} | {time_val:.3f} s"
            )

    print(f"Status: {data.get('pipeline_status', 'N/A')}")
    print(f"Resultado guardado em: {out.resolve()}\n")


def process_input_folder():
    input_folder = PROJECT_ROOT / "input"
    print(f"Procurando vídeos na pasta: {input_folder}")
    if not input_folder.exists():
        raise ValueError(f"A pasta 'input' não foi encontrada em: {input_folder}")

    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    videos = [
        p for p in input_folder.iterdir() if p.suffix.lower() in valid_extensions
    ]
    if not videos:
        print(f"Nenhum vídeo suportado encontrado na pasta '{input_folder}'.")
        return

    print(f"Encontrados {len(videos)} vídeos para processar em input/.\n")
    args_base = arguments()
    for video_file in videos:
        print(f"Processando: {video_file.name}")
        args_base.video = video_file
        args_base.saida = PROJECT_ROOT / "output" / video_file.stem
        try:
            run(args_base)
        except Exception as e:
            print(f"Erro ao processar {video_file.name}: {e}\n")


if __name__ == "__main__":
    print(">>> Iniciando a execução do SpikeAI_rtm...", flush=True)
    args = arguments()
    print(f">>> Argumento 'video' recebido: {args.video}", flush=True)

    try:
        if args.video is None:
            print(
                ">>> Nenhum vídeo especificado. Iniciando varredura da pasta 'input/'...",
                flush=True,
            )
            process_input_folder()
        else:
            print(f">>> Processando vídeo individual: {args.video}", flush=True)
            run(args)
    except ModuleNotFoundError as error:
        print(f">>> Erro de Dependência Ausente: {error.name}", flush=True)
        raise SystemExit(
            f"Dependência ausente: {error.name}. Verifique a venv ativa."
        ) from None
    except (ValueError, OSError, RuntimeError) as error:
        print(f">>> Erro de Execução: {error}", flush=True)
        raise SystemExit(f"Erro: {error}") from None
    except Exception as error:
        print(f">>> Erro Inesperado: {type(error).__name__} - {error}", flush=True)
        raise