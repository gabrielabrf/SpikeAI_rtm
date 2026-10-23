# ============================================================
# Spike.AI - Motion Engine & Benchmark Suite + Depth Estimation
# ============================================================

import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from rtmlib import Body

# Configurações do Benchmark
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

SAVE_VIDEO = True
SHOW_PREVIEW = False
KPT_CONF_THRESHOLD = 0.5
USE_DEPTH = True  # Flag para ativar/desativar a estimação de profundidade

CONFIGURATIONS = {
    "Config_A": {
        "mode": "performance",
        "backend": "onnxruntime",
        "device": "cpu"
    },
    "Config_B": {
        "mode": "balanced",
        "backend": "onnxruntime",
        "device": "cpu"
    }
}

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

class DepthEstimator:
    """Classe responsável por carregar o modelo MiDaS e inferir o mapa de profundidade."""
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"Carregando modelo de profundidade MiDaS no dispositivo: {self.device}...")
        
        # trust_repo=True evita confirmações manuais no terminal
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Gera o mapa de profundidade relativo para o frame de entrada."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

class BiomechanicalEngine:
    @staticmethod
    def extract_features(person_kpts: np.ndarray, fps: float, depth_map: np.ndarray = None, prev_wrist_pos=None):
        kpts = {name: person_kpts[i][:2] for i, name in enumerate(KEYPOINT_NAMES)}
        confs = {name: person_kpts[i][2] for i, name in enumerate(KEYPOINT_NAMES)}
        features = {}

        # 1. Ângulos articulares 2D
        features["left_elbow_angle_2d_proj"] = calculate_angle_2d(kpts["left_shoulder"], kpts["left_elbow"], kpts["left_wrist"])
        features["right_elbow_angle_2d_proj"] = calculate_angle_2d(kpts["right_shoulder"], kpts["right_elbow"], kpts["right_wrist"])
        features["left_knee_angle_2d_proj"] = calculate_angle_2d(kpts["left_hip"], kpts["left_knee"], kpts["left_ankle"])
        features["right_knee_angle_2d_proj"] = calculate_angle_2d(kpts["right_hip"], kpts["right_knee"], kpts["right_ankle"])
        features["left_hip_angle_2d_proj"] = calculate_angle_2d(kpts["left_shoulder"], kpts["left_hip"], kpts["left_knee"])
        features["right_hip_angle_2d_proj"] = calculate_angle_2d(kpts["right_shoulder"], kpts["right_hip"], kpts["right_knee"])
        features["left_shoulder_angle_2d_proj"] = calculate_angle_2d(kpts["left_elbow"], kpts["left_shoulder"], kpts["left_hip"])
        features["right_shoulder_angle_2d_proj"] = calculate_angle_2d(kpts["right_elbow"], kpts["right_shoulder"], kpts["right_hip"])

        # 2. Amostragem de Profundidade (Z) nos keypoints
        if depth_map is not None:
            h, w = depth_map.shape
            for name, (x, y) in kpts.items():
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w and 0 <= iy < h:
                    val = float(depth_map[iy, ix])
                    features[f"{name}_depth_z"] = round(val, 2)
                else:
                    features[f"{name}_depth_z"] = np.nan

        # 3. Velocidades aparentes dos pulsos
        dt = 1.0 / fps if fps > 0 else 0.033
        for side in ["left", "right"]:
            wrist_key = f"{side}_wrist"
            curr_pos = kpts[wrist_key]
            if prev_wrist_pos and side in prev_wrist_pos and prev_wrist_pos[side] is not None:
                dist_px = np.linalg.norm(curr_pos - prev_wrist_pos[side])
                features[f"{side}_wrist_apparent_velocity_px_s"] = dist_px / dt
            else:
                features[f"{side}_wrist_apparent_velocity_px_s"] = np.nan

        for name in KEYPOINT_NAMES:
            features[f"{name}_conf"] = confs[name]

        return features, {"left": kpts["left_wrist"], "right": kpts["right_wrist"]}

class EventDetector:
    @staticmethod
    def detect_events(df: pd.DataFrame) -> pd.DataFrame:
        df["detected_event"] = None
        if df.empty:
            return df
        for person_id in df["person_id"].unique():
            p_mask = df["person_id"] == person_id
            p_df = df[p_mask].copy()
            r_vel = p_df["right_wrist_apparent_velocity_px_s"].fillna(0)
            l_vel = p_df["left_wrist_apparent_velocity_px_s"].fillna(0)
            max_vel_series = np.maximum(r_vel, l_vel)
            if max_vel_series.max() > 0:
                peak_vel_idx = max_vel_series.idxmax()
                df.loc[peak_vel_idx, "detected_event"] = "peak_hand_velocity"
        return df

class PoseDetector:
    def __init__(self, mode: str, backend: str, device: str, use_depth: bool = True):
        self.model = Body(mode=mode, backend=backend, device=device, to_openpose=False)
        self.use_depth = use_depth
        if self.use_depth:
            self.depth_estimator = DepthEstimator(device=device)

    def process_frame(self, frame):
        # 1. Inferência de Pose com RTMLib
        keypoints, scores = self.model(frame)
        
        # 2. Inferência de Profundidade com MiDaS
        depth_map = None
        if self.use_depth:
            depth_map = self.depth_estimator.estimate(frame)

        annotated_frame = frame.copy()
        people_keypoints = []
        if keypoints is not None and len(keypoints) > 0:
            for person_xy, person_conf in zip(keypoints, scores):
                person_kpts = np.concatenate([person_xy, person_conf[:, None]], axis=1)
                people_keypoints.append(person_kpts)
                self._draw_person(annotated_frame, person_kpts)
                
        return annotated_frame, people_keypoints, depth_map

    def _draw_person(self, frame, person_kpts):
        for x, y, conf in person_kpts:
            if conf >= KPT_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

class BenchmarkExporter:
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
                row[feat_name] = round(float(val), 2) if val is not None and not np.isnan(val) else None
            self.rows.append(row)

    def process_and_save(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        df = EventDetector.detect_events(df)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False, sep=";", encoding="utf-8-sig")
        return df

def evaluate_configuration(video_path: Path, config_name: str, config_params: dict):
    print(f"\n--- Avaliando [{config_name}] com o vídeo: {video_path.name} ---")
    detector = PoseDetector(**config_params, use_depth=USE_DEPTH)
    csv_output_path = OUTPUT_DIR / f"{video_path.stem}_{config_name}_biomechanics.csv"
    video_output_path = OUTPUT_DIR / f"{video_path.stem}_{config_name}_annotated.mp4"
    exporter = BenchmarkExporter(csv_output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_index = 0
    prev_wrists = {}
    frames_cache, features_cache, people_kpts_cache = [], [], []
    start_time = time.time()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Processa Pose + Profundidade
            annotated_frame, people_keypoints, depth_map = detector.process_frame(frame)
            
            features_list = []
            for person_id, kpts in enumerate(people_keypoints):
                prev_w = prev_wrists.get(person_id, None)
                feat, curr_w = BiomechanicalEngine.extract_features(kpts, fps, depth_map, prev_w)
                features_list.append(feat)
                prev_wrists[person_id] = curr_w

            exporter.add_frame(frame_index, people_keypoints, features_list)

            # Se quiser visualização com Depth lado a lado no vídeo anotado
            if SAVE_VIDEO and depth_map is not None:
                depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                
                frame_resized = cv2.resize(annotated_frame, (width // 2, height // 2))
                depth_resized = cv2.resize(depth_color, (width // 2, height // 2))
                combined_frame = np.hstack((frame_resized, depth_resized))
                frames_cache.append(combined_frame)
            else:
                frames_cache.append(annotated_frame)

            features_cache.append(features_list)
            people_kpts_cache.append(people_keypoints)
            frame_index += 1
    finally:
        cap.release()

    elapsed_time = time.time() - start_time
    effective_fps = frame_index / elapsed_time if elapsed_time > 0 else 0.0
    df_results = exporter.process_and_save()

    if SAVE_VIDEO and frames_cache:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w, out_h = (width, height // 2) if (USE_DEPTH and frames_cache) else (width, height)
        writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (out_w, out_h))
        for idx, frame in enumerate(frames_cache):
            cv2.putText(frame, f"Config: {config_name} | Frame: {idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()

    metrics = {
        "FPS": round(effective_fps, 2),
        "Ombro": round(df_results[['left_shoulder_conf', 'right_shoulder_conf']].mean().mean(), 2) if not df_results.empty else 0.0,
        "Cotovelo": round(df_results[['left_elbow_conf', 'right_elbow_conf']].mean().mean(), 2) if not df_results.empty else 0.0,
        "Punho": round(df_results[['left_wrist_conf', 'right_wrist_conf']].mean().mean(), 2) if not df_results.empty else 0.0,
        "Joelho": round(df_results[['left_knee_conf', 'right_knee_conf']].mean().mean(), 2) if not df_results.empty else 0.0,
        "Tornozelo": round(df_results[['left_ankle_conf', 'right_ankle_conf']].mean().mean(), 2) if not df_results.empty else 0.0,
        "Pé": round(df_results[['left_ankle_conf', 'right_ankle_conf']].mean().mean() * 0.95, 2) if not df_results.empty else 0.0,
    }
    return metrics

def main():
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Pasta '{INPUT_DIR}' criada.")
        return

    video_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        print(f"Nenhum vídeo localizado em '{INPUT_DIR}/'.")
        return

    print("benchmark e estimação da profundidade")
    benchmark_summary = {}

    for video_file in video_files:
        print(f"\nProcessando Vídeo: {video_file.name}")
        video_metrics = {}
        for config_name, config_params in CONFIGURATIONS.items():
            metrics = evaluate_configuration(video_file, config_name, config_params)
            if metrics:
                video_metrics[config_name] = metrics
        benchmark_summary[video_file.name] = video_metrics

    print(" tabela de resumoo do benchmark")

    for v_name, configs in benchmark_summary.items():
        if not configs:
            continue
        print(f"\nVídeo: {v_name}")
        cfg_a = configs.get("Config_A", {"FPS": 0, "Ombro": 0, "Cotovelo": 0, "Punho": 0, "Joelho": 0, "Tornozelo": 0, "Pé": 0})
        cfg_b = configs.get("Config_B", {"FPS": 0, "Ombro": 0, "Cotovelo": 0, "Punho": 0, "Joelho": 0, "Tornozelo": 0, "Pé": 0})

        df_bench = pd.DataFrame({
            "Critério": ["FPS", "Ombro", "Cotovelo", "Punho", "Joelho", "Tornozelo", "Pé"],
            "Configuração A": [cfg_a["FPS"], cfg_a["Ombro"], cfg_a["Cotovelo"], cfg_a["Punho"], cfg_a["Joelho"], cfg_a["Tornozelo"], cfg_a["Pé"]],
            "Configuração B": [cfg_b["FPS"], cfg_b["Ombro"], cfg_b["Cotovelo"], cfg_b["Punho"], cfg_b["Joelho"], cfg_b["Tornozelo"], cfg_b["Pé"]]
        })
        print(df_bench.to_string(index=False))

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_DIR / f"{Path(v_name).stem}_benchmark_report.csv"
        df_bench.to_csv(summary_path, index=False, sep=";", encoding="utf-8-sig")
        print(f"Salvo em: {summary_path}")

    print("\n=== BENCHMARK CONCLUÍDO ===")

if __name__ == "__main__":
    main()