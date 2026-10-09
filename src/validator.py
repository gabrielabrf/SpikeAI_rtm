import os
import cv2
import json

class SpikeAIValidator:
    def __init__(self, min_width=1280, min_height=720, rec_width=1920, rec_height=1080):
        self.min_width = min_width
        self.min_height = min_height
        self.rec_width = rec_width
        self.rec_height = rec_height

    def validate_video(self, video_path):
        """
        Analisa as propriedades básicas do vídeo (Resolução e FPS)
        e aplica os critérios de aceitação e rejeição automática.
        """
        if not os.path.exists(video_path):
            return {"status": "REJEITAR", "motivo": "Arquivo de vídeo não encontrado."}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "REJEITAR", "motivo": "Não foi possível abrir o arquivo de vídeo."}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        alerts = []
        reasons = []

        # 1. Validação de Resolução (Critério Task 1)
        # Resolução mínima exigida: 720p (1280x720)
        if width < self.min_width or height < self.min_height:
            reasons.append(f"Resolução abaixo do mínimo exigido (720p): {width}x{height}.")
        elif width < self.rec_width or height < self.rec_height:
            alerts.append(f"Resolução abaixo da recomendada (1080p): {width}x{height}[cite: 14].")

        # 2. Validação de Taxa de Quadros (FPS)
        # Mínimo recomendado: 60 FPS (Ideal: 120 FPS para eventos rápidos como impulsão)[cite: 14]
        if fps < 60:
            alerts.append(f"FPS abaixo do recomendado (60 FPS): {fps:.2f} FPS. Risco de erro temporal aumentado[cite: 14].")

        # Resultado da Validação Automática baseada em metadados técnicos
        if reasons:
            return {
                "status": "REJEITAR",
                "resolucao": f"{width}x{height}",
                "fps": round(fps, 2),
                "motivos_rejeicao": reasons,
                "alertas": alerts
            }
        elif alerts:
            return {
                "status": "ACEITAR COM ALERTA",
                "resolucao": f"{width}x{height}",
                "fps": round(fps, 2),
                "motivos_rejeicao": [],
                "alertas": alerts
            }
        else:
            return {
                "status": "ACEITAR",
                "resolucao": f"{width}x{height}",
                "fps": round(fps, 2),
                "motivos_rejeicao": [],
                "alertas": []
            }

    def avaliar_criterios_manuais(self, metadata_manual):
        """
        Avalia critérios visuais que dependem de inspeção ou metadados complementares:
        - Atleta cortado no enquadramento
        - Pés não visíveis no take-off/landing
        - Movimento excessivo de câmera / zoom
        - Oclusão no frame do evento-chave
        - Motion blur excessivo
        """
        status = "ACEITAR"
        reasons = []
        alerts = metadata_manual.get("alertas", [])

        if metadata_manual.get("atleta_cortado", False):
            reasons.append("Atleta cortado no enquadramento do evento-chave[cite: 17].")
        if metadata_manual.get("pes_invisiveis", False):
            reasons.append("Pés não visíveis no take-off/landing[cite: 17].")
        if metadata_manual.get("movimento_excessivo_camera", False):
            reasons.append("Movimento excessivo de câmera ou zoom durante o evento[cite: 17].")
        if metadata_manual.get("oclusao_evento_chave", False):
            reasons.append("Oclusão exata do ponto de contato/evento-chave no frame crítico[cite: 17].")
        if metadata_manual.get("motion_blur_excessivo", False):
            reasons.append("Motion blur excessivo no frame do evento[cite: 17].")
        if metadata_manual.get("iluminacao_insuficiente", False):
            reasons.append("Iluminação insuficiente[cite: 17].")

        if reasons:
            return {"status": "REJEITAR", "motivos": reasons}
        
        if metadata_manual.get("distancia_excessiva", False):
            alerts.append("Distância excessiva (atleta ocupa menos de um terço do frame).")
        if metadata_manual.get("contraluz", False):
            alerts.append("Presença de contraluz ou reflexo forte[cite: 17, 18].")

        if alerts:
            return {"status": "ACEITAR COM ALERTA", "alertas": alerts}

        return {"status": "ACEITAR", "motivos": [], "alertas": []}

# Exemplo de Uso
if __name__ == "__main__":
    validator = SpikeAIValidator()
    
    # Exemplo apontando para um vídeo de entrada do dataset
    video_exemplo = "SpikeAI_rtm-main/input/ataque_braço.mp4"
    
    resultado_tecnico = validator.validate_video(video_exemplo)
    print("--- Resultado da Validação Técnica ---")
    print(json.dumps(resultado_tecnico, indent=4, ensure_ascii=False))