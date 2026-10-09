import json

class SpikeAIGroundTruthManager:
    def __init__(self):
        # Escopo oficial do MVP (Task 2)
        self.mvp_events = {
            "take_off": {
                "prioridade": "Essencial (MVP)",
                "descricao": "Último frame com contato visível do atleta com o solo antes da fase aérea."
            },
            "landing": {
                "prioridade": "Essencial (MVP)",
                "descricao": "Primeiro frame com contato visível de qualquer parte do pé com o solo após a fase aérea."
            },
            "inicio_backswing": {
                "prioridade": "Essencial (MVP)",
                "descricao": "Primeiro frame em que a mão do braço de ataque inicia deslocamento para trás[cite: 23.2]."
            },
            "cocking": {
                "prioridade": "Essencial (MVP)",
                "descricao": "Frame em que o braço atinge sua extensão máxima antes da fase de aceleração[cite: 23.2]."
            }
        }
        
        # Eventos secundários e adiáveis mantidos fora do escopo primário do MVP, 
        # mas estruturados para futuras versões.
        self.secondary_events = ["plant", "inicio_push_off", "peak_hand_velocity"]
        self.deferred_events = ["countermovement", "pico_do_salto", "inicio_desaceleracao"]

    def registrar_evento_ground_truth(self, event_name, frame_number, uncertainty_margin=0):
        """
        Registra um evento essencial do MVP garantindo a estrutura JSON 
        com o frame provável e a margem de incerteza (Task 4).
        """
        if event_name not in self.mvp_events:
            raise ValueError(f"O evento '{event_name}' não faz parte do escopo prioritário do MVP.")

        registro = {
            "event": event_name,
            "frame": frame_number,
            "uncertainty": uncertainty_margin,
            "metadata_evento": self.mvp_events[event_name]
        }
        
        return registro

# Exemplo de uso para exportação de anotações Ground Truth em JSON
if __name__ == "__main__":
    manager = SpikeAIGroundTruthManager()
    
    # Exemplo de anotação de um vídeo de ataque
    anotacoes_video = [
        manager.registrar_evento_ground_truth("inicio_backswing", frame_number=110, uncertainty_margin=1),
        manager.registrar_evento_ground_truth("take_off", frame_number=142, uncertainty_margin=2),
        manager.registrar_evento_ground_truth("landing", frame_number=185, uncertainty_margin=0)
    ]
    
    print("--- Ground Truth JSON (MVP Scope) ---")
    print(json.dumps(anotacoes_video, indent=4, ensure_ascii=False))