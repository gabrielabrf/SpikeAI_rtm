"""
Spike.AI - Motion Engine (Task 3 da Issue 6)
Pipeline de Pose Estimation com RTMPose (via rtmlib) para Análise Biomecânica de Voleibol.

Métricas 2D Calculadas e Exibidas no Vídeo:
  - Visualização dos Ângulos Articulares (Cotovelo, Ombro, Joelho).
  - Painel de métricas na tela (Frame, Velocidade do Punho).
  - Exportação estruturada para CSV e TXT Formatado usando Pandas.

Requisitos:
    pip install rtmlib opencv-python opencv-contrib-python numpy onnxruntime pandas
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from rtmlib import Body


# ============================================================
# CONFIGURAÇÕES DA TASK 3
# ============================================================

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

SAVE_VIDEO = True
SHOW_PREVIEW = False

POSE_MODE = "performance"
DEVICE = "cpu"
BACKEND = "onnxruntime"

KPT_CONF_THRESHOLD = 0.5

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # braços/ombros
    (5, 11), (6, 12), (11, 12),                      # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),          # pernas
    (0, 1), (0, 2), (1, 3), (2, 4),                  # rosto
]


# ============================================================
# MOTOR BIOMECÂNICO 2D
# ============================================================

def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calcula o ângulo 2D em graus entre três pontos (A -> B -> C) onde B é o vértice."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


class BiomechanicalEngine:
    """Extrai as features derivadas e anota métricas visuais no vídeo."""

    @staticmethod
    def extract_features(person_kpts: np.ndarray, fps: float, prev_wrist_pos=None):
        kpts = {name: person_kpts[i][:2] for i, name in enumerate(KEYPOINT_NAMES)}

        features = {}

        # 1. Ângulos
        features["left_elbow_angle"] = calculate_angle_2d(kpts["left_shoulder"], kpts["left_elbow"], kpts["left_wrist"])
        features["right_elbow_angle"] = calculate_angle_2d(kpts["right_shoulder"], kpts["right_elbow"], kpts["right_wrist"])

        features["left_knee_angle"] = calculate_angle_2d(kpts["left_hip"], kpts["left_knee"], kpts["left_ankle"])
        features["right_knee_angle"] = calculate_angle_2d(kpts["right_hip"], kpts["right_knee"], kpts["right_ankle"])

        features["left_hip_angle"] = calculate_angle_2d(kpts["left_shoulder"], kpts["left_hip"], kpts["left_knee"])
        features["right_hip_angle"] = calculate_angle_2d(kpts["right_shoulder"], kpts["right_hip"], kpts["right_knee"])

        features["left_shoulder_angle"] = calculate_angle_2d(kpts["left_elbow"], kpts["left_shoulder"], kpts["left_hip"])
        features["right_shoulder_angle"] = calculate_angle_2d(kpts["right_elbow"], kpts["right_shoulder"], kpts["right_hip"])

        # 2. Normalização
        mid_ankle_y = (kpts["left_ankle"][1] + kpts["right_ankle"][1]) / 2.0
        body_height_px = abs(mid_ankle_y - kpts["nose"][1])
        features["body_height_px"] = body_height_px if body_height_px > 0 else np.nan

        # 3. Velocidade Aparente do Punho
        dt = 1.0 / fps if fps > 0 else 0.033
        for side in ["left", "right"]:
            wrist_key = f"{side}_wrist"
            curr_pos = kpts[wrist_key]

            if prev_wrist_pos and side in prev_wrist_pos and prev_wrist_pos[side] is not None:
                dist_px = np.linalg.norm(curr_pos - prev_wrist_pos[side])
                features[f"{side}_wrist_velocity_px_s"] = dist_px / dt
            else:
                features[f"{side}_wrist_velocity_px_s"] = np.nan

        return features, {"left": kpts["left_wrist"], "right": kpts["right_wrist"]}

    @staticmethod
    def draw_annotations(frame: np.ndarray, person_kpts: np.ndarray, features: dict):
        """Desenha os ângulos calculados diretamente sobre as articulações no vídeo."""
        kpts = {name: person_kpts[i] for i, name in enumerate(KEYPOINT_NAMES)}

        # Lista de articulações para exibir texto de ângulo no frame
        angles_to_draw = [
            ("right_elbow", features.get("right_elbow_angle")),
            ("left_elbow", features.get("left_elbow_angle")),
            ("right_knee", features.get("right_knee_angle")),
            ("left_knee", features.get("left_knee_angle")),
        ]

        for kpt_name, angle_val in angles_to_draw:
            if angle_val is not None and not np.isnan(angle_val):
                pt = kpts[kpt_name]
                conf = pt[2]
                if conf >= KPT_CONF_THRESHOLD:
                    x, y = int(pt[0]), int(pt[1])
                    text = f"{int(angle_val)}deg"
                    # Desenha um fundo escuro leve no texto para destacar
                    cv2.putText(frame, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                    cv2.putText(frame, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


# ============================================================
# DETECTOR DE POSE
# ============================================================

class PoseDetector:
    def __init__(self, mode: str, backend: str, device: str):
        self.model = Body(mode=mode, backend=backend, device=device, to_openpose=False)

    def process_frame(self, frame):
        keypoints, scores = self.model(frame)
        annotated_frame = frame.copy()
        people_keypoints = []

        if keypoints is not None and len(keypoints) > 0:
            for person_xy, person_conf in zip(keypoints, scores):
                person_kpts = np.concatenate([person_xy, person_conf[:, None]], axis=1)
                people_keypoints.append(person_kpts)
                self._draw_person(annotated_frame, person_kpts)

        return annotated_frame, people_keypoints

    def _draw_person(self, frame, person_kpts):
        for x, y, conf in person_kpts:
            if conf >= KPT_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        for i, j in SKELETON_CONNECTIONS:
            xi, yi, ci = person_kpts[i]
            xj, yj, cj = person_kpts[j]
            if ci >= KPT_CONF_THRESHOLD and cj >= KPT_CONF_THRESHOLD:
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 200, 255), 2)


# ============================================================
# EXPORTADOR CSV/TXT ORGANIZADO (PANDAS)
# ============================================================

class CSVExporter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.rows = []

    def add_frame(self, frame_index: int, people_keypoints, features_list):
        for person_id, (kpts, feat) in enumerate(zip(people_keypoints, features_list)):
            row = {"frame": frame_index, "person_id": person_id}
            
            for name, (x, y, conf) in zip(KEYPOINT_NAMES, kpts):
                row[f"{name}_x"] = round(float(x), 2)
                row[f"{name}_y"] = round(float(y), 2)
                row[f"{name}_conf"] = round(float(conf), 2)
            
            for feat_name, val in feat.items():
                row[feat_name] = round(float(val), 2) if not np.isnan(val) else None

            self.rows.append(row)

    def save(self):
        if not self.rows:
            print("Nenhum dado coletado para exportação.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)

        # Exporta o CSV padrão
        df.to_csv(self.output_path, index=False, sep=",", encoding="utf-8-sig")

        # Exporta uma versão em arquivo de texto com colunas alinhadas por espaço
        txt_output_path = self.output_path.with_suffix(".txt")
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))

        print(f"CSV salvo em: {self.output_path}")
        print(f"Tabela de texto alinhada salva em: {txt_output_path} ({len(df)} linhas)")


# ============================================================
# PROCESSADOR DE VÍDEO
# ============================================================

def process_single_video(video_path: Path, detector: PoseDetector):
    print(f"\n========================================")
    print(f"Analisando biomecânica: {video_path.name}")
    print(f"========================================")

    csv_output_path = OUTPUT_DIR / f"{video_path.stem}_biomechanics.csv"
    video_output_path = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"

    exporter = CSVExporter(csv_output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_VIDEO:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

    frame_index = 0
    prev_wrists = {}
    start_time = time.time()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, people_keypoints = detector.process_frame(frame)
            
            features_list = []
            for person_id, kpts in enumerate(people_keypoints):
                prev_w = prev_wrists.get(person_id, None)
                feat, curr_w = BiomechanicalEngine.extract_features(kpts, fps, prev_w)
                features_list.append(feat)
                prev_wrists[person_id] = curr_w

                # Desenha os ângulos sob as articulações no vídeo anotado
                BiomechanicalEngine.draw_annotations(annotated_frame, kpts, feat)

            # Painel HUD com informações do vídeo
            cv2.putText(annotated_frame, f"Frame: {frame_index}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Spike.AI - Motion Engine", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            exporter.add_frame(frame_index, people_keypoints, features_list)

            if writer is not None:
                writer.write(annotated_frame)

            if SHOW_PREVIEW:
                cv2.imshow("Spike.AI Biomechanics", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

    finally:
        elapsed_time = time.time() - start_time
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        exporter.save()

        print(f"Processamento concluído em {elapsed_time:.2f}s | Total de frames: {frame_index}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Pasta '{INPUT_DIR}' criada. Coloque os vídeos lá dentro e execute.")
        return

    video_files = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"Nenhum vídeo localizado na pasta '{INPUT_DIR}/'.")
        return

    print(f"Localizados {len(video_files)} vídeos. Iniciando RTMPose (Task 3)")
    detector = PoseDetector(mode=POSE_MODE, backend=BACKEND, device=DEVICE)

    for video_file in video_files:
        process_single_video(video_file, detector)


if __name__ == "__main__":
    main()