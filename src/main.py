"""
Spike.AI - Motion Engine (Task 3 da Issue 6)
Pipeline de Pose Estimation com RTMPose (via rtmlib) para Análise Biomecânica de Voleibol.

Métricas 2D Calculadas e Exibidas no Vídeo:
  - Visualização dos Ângulos Articulares Projetados em 2D.
  - Painel de métricas na tela (Frame, Velocidade Aparente do Punho).
  - Detecção Automática de Eventos Biomecânicos (Peak Velocity, Cocking, Salto, Aterrissagem).
  - Exportação estruturada para CSV e TXT Formatado com Nomenclatura Explícita 2D.

NOTA DA BIOMECÂNICA 2D:
  - As métricas calculadas correspondem a projeções no plano 2D da imagem.
  - Ângulos (*_angle_2d_proj): Ângulos planos formados no plano da câmera (sem compensação de profundidade 3D).
  - Velocidades (*_apparent_velocity_px_s): Velocidade escalar aparente medida no plano em pixels/segundo.

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
    """Calcula o ângulo plano projetado em 2D (em graus) entre três pontos (A -> B -> C) no plano da imagem."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


class BiomechanicalEngine:
    """Extrai as features derivadas com nomenclatura explícita de projeção 2D."""

    @staticmethod
    def extract_features(person_kpts: np.ndarray, fps: float, prev_wrist_pos=None):
        kpts = {name: person_kpts[i][:2] for i, name in enumerate(KEYPOINT_NAMES)}

        features = {}

        # 1. Ângulos Planos Projetados em 2D (Especificação Explícita de Projeção)
        features["left_elbow_angle_2d_proj"] = calculate_angle_2d(kpts["left_shoulder"], kpts["left_elbow"], kpts["left_wrist"])
        features["right_elbow_angle_2d_proj"] = calculate_angle_2d(kpts["right_shoulder"], kpts["right_elbow"], kpts["right_wrist"])

        features["left_knee_angle_2d_proj"] = calculate_angle_2d(kpts["left_hip"], kpts["left_knee"], kpts["left_ankle"])
        features["right_knee_angle_2d_proj"] = calculate_angle_2d(kpts["right_hip"], kpts["right_knee"], kpts["right_ankle"])

        features["left_hip_angle_2d_proj"] = calculate_angle_2d(kpts["left_shoulder"], kpts["left_hip"], kpts["left_knee"])
        features["right_hip_angle_2d_proj"] = calculate_angle_2d(kpts["right_shoulder"], kpts["right_hip"], kpts["right_knee"])

        features["left_shoulder_angle_2d_proj"] = calculate_angle_2d(kpts["left_elbow"], kpts["left_shoulder"], kpts["left_hip"])
        features["right_shoulder_angle_2d_proj"] = calculate_angle_2d(kpts["right_elbow"], kpts["right_shoulder"], kpts["right_hip"])

        # 2. Atendendo explicitamente à provocação da Task 3: Rotação / Inclinação Aparente do Ombro em 2D
        # (Calculada como o ângulo de elevação do úmero em relação à linha vertical do tronco)
        def apparent_shoulder_rotation_2d(shoulder, elbow, hip):
            if np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(hip)):
                return np.nan
            # Vetor tronco (Hip -> Shoulder) e Vetor Braço/Úmero (Shoulder -> Elbow)
            trunk_vec = shoulder - hip
            arm_vec = elbow - shoulder
            cosine = np.dot(trunk_vec, arm_vec) / (np.linalg.norm(trunk_vec) * np.linalg.norm(arm_vec) + 1e-6)
            return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

        features["left_apparent_shoulder_rotation_2d_deg"] = apparent_shoulder_rotation_2d(kpts["left_shoulder"], kpts["left_elbow"], kpts["left_hip"])
        features["right_apparent_shoulder_rotation_2d_deg"] = apparent_shoulder_rotation_2d(kpts["right_shoulder"], kpts["right_elbow"], kpts["right_hip"])

        # 3. Normalização de Escala (Altura Aparente em Pixels no Plano Focal)
        mid_ankle_y = (kpts["left_ankle"][1] + kpts["right_ankle"][1]) / 2.0
        body_height_px = abs(mid_ankle_y - kpts["nose"][1])
        features["body_height_px"] = body_height_px if body_height_px > 0 else np.nan

        # 4. Velocidade Aparente do Punho (px/s no Plano da Câmera)
        dt = 1.0 / fps if fps > 0 else 0.033
        for side in ["left", "right"]:
            wrist_key = f"{side}_wrist"
            curr_pos = kpts[wrist_key]

            if prev_wrist_pos and side in prev_wrist_pos and prev_wrist_pos[side] is not None:
                dist_px = np.linalg.norm(curr_pos - prev_wrist_pos[side])
                features[f"{side}_wrist_apparent_velocity_px_s"] = dist_px / dt
            else:
                features[f"{side}_wrist_apparent_velocity_px_s"] = np.nan

        return features, {"left": kpts["left_wrist"], "right": kpts["right_wrist"]}


# ============================================================
# DETECTOR DE EVENTOS BIOMECÂNICOS
# ============================================================

class EventDetector:
    """Analisador de séries temporais para detecção automática de eventos motores."""

    @staticmethod
    def detect_events(df: pd.DataFrame) -> pd.DataFrame:
        df["detected_event"] = None

        if df.empty:
            return df

        for person_id in df["person_id"].unique():
            p_mask = df["person_id"] == person_id
            p_df = df[p_mask].copy()

            # 1. Peak Hand Velocity
            r_vel = p_df["right_wrist_apparent_velocity_px_s"].fillna(0)
            l_vel = p_df["left_wrist_apparent_velocity_px_s"].fillna(0)
            max_vel_series = np.maximum(r_vel, l_vel)

            if max_vel_series.max() > 0:
                peak_vel_idx = max_vel_series.idxmax()
                df.loc[peak_vel_idx, "detected_event"] = "peak_hand_velocity"

                # 2. Cocking / Backswing (Antes do pico de velocidade)
                pre_peak_df = p_df.loc[:peak_vel_idx]
                if not pre_peak_df.empty:
                    max_elbow_idx = pre_peak_df["right_elbow_angle_2d_proj"].idxmax()
                    if pd.notna(max_elbow_idx):
                        df.loc[max_elbow_idx, "detected_event"] = "cocking_backswing"

            # 3. Análise da Fase Aérea (Salto, Take-off e Landing)
            ankles_y = (p_df["left_ankle_y"] + p_df["right_ankle_y"]) / 2.0
            
            # Altura base do solo (maior Y representa os pés no chão)
            base_ground_y = ankles_y.quantile(0.8)
            
            # Ponto de altura máxima do salto (menor Y)
            apex_idx = ankles_y.idxmin()
            max_jump_height = base_ground_y - ankles_y.loc[apex_idx]

            # Se houve salto significativo (> 30px de variação vertical)
            if max_jump_height > 30:
                df.loc[apex_idx, "detected_event"] = "jump_peak"

                # Take-off: último frame no chão antes do ápice do salto
                pre_apex = ankles_y.loc[:apex_idx]
                takeoff_series = pre_apex[pre_apex >= base_ground_y - 10]
                if not takeoff_series.empty:
                    df.loc[takeoff_series.index[-1], "detected_event"] = "take_off"

                # Landing: primeiro frame ao retornar à altura base após o ápice
                post_apex = ankles_y.loc[apex_idx:]
                landing_series = post_apex[post_apex >= base_ground_y - 10]
                if not landing_series.empty:
                    df.loc[landing_series.index[0], "detected_event"] = "landing"

        return df


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
# EXPORTADOR CSV/TXT ORGANIZADO (MONOSPACED TABLE)
# ============================================================

class CSVExporter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.rows = []

    def add_frame(self, frame_index: int, people_keypoints, features_list):
        for person_id, (kpts, feat) in enumerate(zip(people_keypoints, features_list)):
            row = {"frame": frame_index, "person_id": person_id}
            
            # Keypoints
            for name, (x, y, conf) in zip(KEYPOINT_NAMES, kpts):
                row[f"{name}_x"] = round(float(x), 2)
                row[f"{name}_y"] = round(float(y), 2)
                row[f"{name}_conf"] = round(float(conf), 2)
            
            # Features biomecânicas (com a nova nomenclatura 2D explicitada)
            for feat_name, val in feat.items():
                row[feat_name] = round(float(val), 2) if not np.isnan(val) else None

            self.rows.append(row)

    def _export_monospaced_txt_table(self, df: pd.DataFrame, txt_path: Path):
        """
        Gera uma tabela com espaçamento fixo (Monospaced Grid Table).
        Contém explicitamente a notas técnicas sobre a natureza projetada (2D) dos dados.
        """
        formatted_df = df.fillna("-")
        str_df = formatted_df.astype(str)

        col_widths = {}
        for col in str_df.columns:
            max_len = max(str_df[col].apply(len).max(), len(col))
            col_widths[col] = max_len + 3

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 90 + "\n")
                f.write(" SPIKE.AI - TABELA DE DADOS BIOMECÂNICOS 2D (SÉRIE TEMPORAL)\n")
                f.write(" NOTA TÉCNICA: Os ângulos (*_angle_2d_proj) são projeções no plano da imagem.\n")
                f.write("               Velocidades (*_apparent_velocity_px_s) são relativas à escala em pixels.\n")
                f.write("=" * 90 + "\n\n")

                header_str = "".join([f"{col:<{col_widths[col]}}" for col in str_df.columns])
                f.write(header_str + "\n")

                separator_str = "".join(["-" * (col_widths[col] - 1) + " " for col in str_df.columns])
                f.write(separator_str + "\n")

                for _, row in str_df.iterrows():
                    row_str = "".join([f"{row[col]:<{col_widths[col]}}" for col in str_df.columns])
                    f.write(row_str + "\n")

                f.write("\n" + "=" * 90 + "\n")
                f.write(f" Total de Registros Exportados: {len(df)} | Colunas: {len(df.columns)}\n")

            print(f"Tabela TXT Alinhada salva em: {txt_path}")
        except PermissionError:
            print(f"\n[ERRO DE PERMISSÃO] Não foi possível gravar em '{txt_path}'. Feche o arquivo TXT se estiver aberto.")

    def process_and_save(self) -> pd.DataFrame:
        if not self.rows:
            print("Nenhum dado coletado para exportação.")
            return pd.DataFrame()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)

        # Aplica a detecção automática de eventos
        df = EventDetector.detect_events(df)

        # 1. Salva CSV com separador ';' para Excel
        try:
            df.to_csv(self.output_path, index=False, sep=";", encoding="utf-8-sig")
            print(f"CSV Bruto salvo em        : {self.output_path}")
        except PermissionError:
            print(f"\n[ERRO DE PERMISSÃO] O arquivo '{self.output_path}' está aberto no Excel!")
            print("Por favor, feche o Excel e execute o script novamente para atualizar o arquivo.\n")

        # 2. Salva Tabela Monospaced no TXT (para leitura no VS Code/Bloco de Notas)
        txt_output_path = self.output_path.with_suffix(".txt")
        self._export_monospaced_txt_table(df, txt_output_path)

        return df


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

    frame_index = 0
    prev_wrists = {}
    frames_cache = []
    features_cache = []
    people_kpts_cache = []
    start_time = time.time()

    # Passagem 1: Extrair features e ler todos os frames
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

            exporter.add_frame(frame_index, people_keypoints, features_list)
            
            frames_cache.append(annotated_frame)
            features_cache.append(features_list)
            people_kpts_cache.append(people_keypoints)

            frame_index += 1

    finally:
        cap.release()

    # Processa os eventos no DataFrame completo e salva TXT/CSV
    df_events = exporter.process_and_save()

    # Passagem 2: Renderizar vídeo final (Limpo)
    if SAVE_VIDEO and frames_cache:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

        for idx, (frame, features_list, people_keypoints) in enumerate(zip(frames_cache, features_cache, people_kpts_cache)):
            # Evento detectado para exibição visual
            event_name = None
            if not df_events.empty:
                frame_events = df_events[df_events["frame"] == idx]["detected_event"].dropna()
                if not frame_events.empty:
                    event_name = frame_events.iloc[0]

            # Exibe apenas o evento na tela se ele ocorrer (sem os ângulos nas articulações)
            if event_name:
                cv2.putText(frame, f"EVENTO: {event_name.upper()}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Exibe apenas o contador do número do Frame
            cv2.putText(frame, f"Frame: {idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()

    elapsed_time = time.time() - start_time
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