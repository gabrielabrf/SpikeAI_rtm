import cv2
import json
import numpy as np
from typing import Dict, Any, List, Optional


class SpikeArmAnalyzer:
    """
    Módulo responsável por analisar as fases do braço no ataque do vôlei
    e integrar com os eventos das tasks #13 e #15.
    """
    def __init__(self, video_path: str, config: Optional[Dict[str, Any]] = None):
        self.video_path = video_path
        self.config = config or {}
        
        # Dados recebidos de integrações anteriores (#13 e #15)
        self.feature_13_data = self.config.get("integration_13_data", {})
        self.feature_15_data = self.config.get("integration_15_data", {})
        
        # Estrutura padrão para os 4 eventos do ataque com suporte a INCONCLUSIVO
        self.events = {
            "preparacao_inicio": {
                "frame": None, "time": None, "status": "INCONCLUSIVO"
            },
            "preparacao_maxima_aparente": {
                "frame": None, "time": None, "status": "INCONCLUSIVO"
            },
            "braco_frente_inicio": {
                "frame": None, "time": None, "status": "INCONCLUSIVO"
            },
            "saida_vs_apice": {
                "frame": None, "time": None, "status": "INCONCLUSIVO", "sub_type": None
            }
        }

    def _apply_trio_rules(self, keypoints_sequence: List[Dict[str, Any]], fps: float) -> Dict[str, Any]:
        """
        Aplica as regras do trio para detectar a Preparação Máxima Aparente
        e o Início do Braço para Frente.
        """
        prep_max_frame = None
        arm_forward_frame = None
        
        for idx, kp in enumerate(keypoints_sequence):
            frame_num = kp.get("frame", idx)
            angle = kp.get("elbow_shoulder_angle", 0)
            velocity = kp.get("arm_velocity", 0)
            
            # Regra Trio 1: Pico de extensão/ângulo do cotovelo/ombro (Preparação Máxima)
            if angle > 135 and prep_max_frame is None:
                prep_max_frame = frame_num
                
            # Regra Trio 2: Mudança do vetor de aceleração/velocidade para frente
            if prep_max_frame is not None and frame_num > prep_max_frame and velocity > 0.5:
                arm_forward_frame = frame_num
                break

        results = {}
        
        # Validação do evento: Preparação Máxima
        if prep_max_frame is not None:
            results["preparacao_maxima_aparente"] = {
                "frame": prep_max_frame,
                "time": round(prep_max_frame / fps, 3),
                "status": "VALIDO"
            }
        else:
            results["preparacao_maxima_aparente"] = {
                "frame": None, "time": None, "status": "INCONCLUSIVO"
            }

        # Validação do evento: Início do Braço para Frente
        if arm_forward_frame is not None:
            results["braco_frente_inicio"] = {
                "frame": arm_forward_frame,
                "time": round(arm_forward_frame / fps, 3),
                "status": "VALIDO"
            }
        else:
            results["braco_frente_inicio"] = {
                "frame": None, "time": None, "status": "INCONCLUSIVO"
            }

        return results

    def _distinguish_saida_apice(self, frame_data: Dict[str, Any], fps: float) -> Dict[str, Any]:
        """
        Diferencia claramente a 'Saída' (início da fase aérea/impulso) 
        do 'Ápice' (ponto mais alto do salto).
        """
        frame = frame_data.get("frame")
        event_type = str(frame_data.get("detected_type", "")).lower()
        
        if frame is not None and event_type in ["saida", "apice"]:
            return {
                "frame": frame,
                "time": round(frame / fps, 3),
                "status": "VALIDO",
                "sub_type": event_type  # 'saida' ou 'apice'
            }
        
        return {
            "frame": None,
            "time": None,
            "status": "INCONCLUSIVO",
            "sub_type": None
        }

    def process_attack(self, keypoints_data: List[Dict[str, Any]], output_video_path: str, output_json_path: str) -> Dict[str, Any]:
        """
        Processa os 4 eventos, renderiza o vídeo anotado e salva o JSON integrado.
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 1. Aplicação das Regras do Trio
        trio_results = self._apply_trio_rules(keypoints_data, fps)
        self.events["preparacao_maxima_aparente"] = trio_results["preparacao_maxima_aparente"]
        self.events["braco_frente_inicio"] = trio_results["braco_frente_inicio"]

        # 2. Integração com a Task #13 (Início da Preparação)
        if "preparacao_inicio_frame" in self.feature_13_data:
            f = self.feature_13_data["preparacao_inicio_frame"]
            if f is not None:
                self.events["preparacao_inicio"] = {
                    "frame": f,
                    "time": round(f / fps, 3),
                    "status": "VALIDO"
                }

        # 3. Integração com a Task #15 (Saída vs Ápice)
        self.events["saida_vs_apice"] = self._distinguish_saida_apice(self.feature_15_data, fps)

        # 4. Renderização com Overlays dos Eventos sem perder a taxa de quadros
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            y_offset = 30
            for event_name, data in self.events.items():
                if data["status"] == "VALIDO":
                    label = f"{event_name.upper()}: Frame {data['frame']} ({data['time']}s)"
                    if data.get("sub_type"):
                        label += f" [{data['sub_type'].upper()}]"
                    color = (0, 255, 0) # Verde para eventos detectados
                else:
                    label = f"{event_name.upper()}: INCONCLUSIVO"
                    color = (0, 0, 255) # Vermelho para inconclusivo

                cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y_offset += 25

            out.write(frame)

        cap.release()
        out.release()

        # 5. Estruturação e Exportação do Relatório Integrado
        output_payload = {
            "video_source": self.video_path,
            "fps": fps,
            "events_summary": self.events,
            "integrations": {
                "task_13_data": self.feature_13_data,
                "task_15_data": self.feature_15_data
            }
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_payload, f, indent=4, ensure_ascii=False)

        return output_payload