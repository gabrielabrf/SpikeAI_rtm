import json

class SpikeAIExportHandler:
    def __init__(self, max_uncertainty_threshold=3):
        # Limite máximo de quadros de incerteza tolerado antes de exigir revisão manual ou descarte
        self.max_uncertainty_threshold = max_uncertainty_threshold

    def processar_e_exportar(self, video_id, eventos_detectados):
        """
        Processa os eventos detectados, calcula a incerteza, aplica regras
        de rejeição por ambiguidade e gera a estrutura JSON final.
        """
        eventos_processados = []
        status_geral = "APROVADO"
        alertas_criticos = []

        for ev in eventos_detectados:
            nome_evento = ev.get("event")
            frame_provavel = ev.get("frame")
            incerteza = ev.get("uncertainty", 0)

            # Regra de tratamento para ambiguidades elevadas[cite: 21]
            if incerteza > self.max_uncertainty_threshold:
                status_evento = "AMBIGUO_REQUER_REVISAO"
                status_geral = "PENDENTE_REVISAO_MANUAL"
                alertas_criticos.append(
                    f"Incerteza alta no evento '{nome_evento}' (margem de {incerteza} quadros excede o limite de {self.max_uncertainty_threshold})."
                )
            else:
                status_evento = "VALIDO"

            evento_formatado = {
                "event": nome_evento,
                "frame": frame_provavel,
                "uncertainty": incerteza,
                "status": status_evento
            }
            eventos_processados.append(evento_formatado)

        # Estrutura final de exportação em JSON
        output_data = {
            "video_id": video_id,
            "pipeline_status": status_geral,
            "alertas": alertas_criticos,
            "ground_truth_mvp": eventos_processados
        }

        return json.dumps(output_data, indent=4, ensure_ascii=False)

# Exemplo prático de uso da Task 4
if __name__ == "__main__":
    handler = SpikeAIExportHandler(max_uncertainty_threshold=2)

    # Simulação de eventos detectados com diferentes margens de incerteza
    eventos_brutos = [
        {"event": "inicio_backswing", "frame": 110, "uncertainty": 1},
        {"event": "take_off", "frame": 142, "uncertainty": 4},  # Ambiguidade alta (excede o limite)
        {"event": "landing", "frame": 185, "uncertainty": 0}
    ]

    json_exportado = handler.processar_e_exportar("video_ataque_01", eventos_brutos)
    print("--- JSON de Exportação Oficial (Task 4) ---")
    print(json_exportado)