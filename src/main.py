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
    """Integra a detecção dos 4 eventos do ataque de vôlei com regras do trio,
    distinção entre Saída e Ápice e suporte a resultados INCONCLUSIVO.
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

    def _extract_arm_metrics(self) -> List[Dict[str, Any]]:
        metrics = []
        for idx, pose in enumerate(self.poses):
            r_shoulder, r_elbow, r_wrist = pose[2], pose[3], pose[4]
            angle = 0.0
            velocity = 0.0

            if r_shoulder[2] > 0.2 and r_elbow[2] > 0.2 and r_wrist[2] > 0.2:
                v1 = r_shoulder[:2] - r_elbow[:2]
                v2 = r_wrist[:2] - r_elbow[:2]
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)

                if norm_v1 > 0 and norm_v2 > 0:
                    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

                if idx > 0 and self.poses[idx - 1][4][2] > 0.2:
                    dt = self.timestamps[idx] - self.timestamps[idx - 1]
                    if dt > 0:
                        dist = np.linalg.norm(
                            r_wrist[:2] - self.poses[idx - 1][4][:2]
                        )
                        velocity = dist / dt

            metrics.append(
                {"frame": idx, "angle": float(angle), "velocity": float(velocity)}
            )
        return metrics

    def process_arm_events(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        metrics = self._extract_arm_metrics()
        prep_inicio = None
        saida_apice = None

        for ev in self.events_base:
            if ev.get("event") in ["take_off", "preparacao_inicio"]:
                prep_inicio = ev
                break

        prep_max_frame = None
        for item in metrics:
            if item["angle"] > 135 and prep_max_frame is None:
                prep_max_frame = item["frame"]
                break

        braco_frente_frame = None
        if prep_max_frame is not None:
            for item in metrics[prep_max_frame:]:
                if item["velocity"] > 150.0:
                    braco_frente_frame = item["frame"]
                    break

        for ev in self.events_base:
            if ev.get("event") in ["landing", "apice", "saida"]:
                ev_type = ev.get(
                    "sub_type",
                    "saida" if ev.get("event") == "take_off" else "apice",
                )
                saida_apice = {
                    "event": "saida_vs_apice",
                    "status": ev.get("status", "VALIDO"),
                    "frame": ev.get("frame"),
                    "time_s": ev.get("time_s"),
                    "sub_type": ev_type,
                    "reason": ev.get("reason", ""),
                }
                break

        final_events = [
            prep_inicio
            or {
                "event": "preparacao_inicio",
                "status": "INCONCLUSIVO",
                "frame": None,
                "time_s": None,
                "reason": "Não detectado pelo módulo #13",
            },
            {
                "event": "preparacao_maxima_aparente",
                "status": "VALIDO" if prep_max_frame is not None else "INCONCLUSIVO",
                "frame": prep_max_frame,
                "time_s": float(self.timestamps[prep_max_frame])
                if prep_max_frame is not None
                else None,
                "reason": ""
                if prep_max_frame is not None
                else "Pico de ângulo de extensão não atingido",
            },
            {
                "event": "braco_frente_inicio",
                "status": "VALIDO" if braco_frente_frame is not None else "INCONCLUSIVO",
                "frame": braco_frente_frame,
                "time_s": float(self.timestamps[braco_frente_frame])
                if braco_frente_frame is not None
                else None,
                "reason": ""
                if braco_frente_frame is not None
                else "Aceleração do braço não detectada",
            },
            saida_apice
            or {
                "event": "saida_vs_apice",
                "status": "INCONCLUSIVO",
                "frame": None,
                "time_s": None,
                "sub_type": None,
                "reason": "Não detectado pelo módulo #15",
            },
        ]

        summary = {ev["event"]: ev for ev in final_events}
        return final_events, summary


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