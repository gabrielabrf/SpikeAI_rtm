"""Validação técnica do vídeo de entrada, antes de qualquer processamento."""

from pathlib import Path

import av

RECOMMENDED_FPS = 60


def validate_video(path):
    """ACEITAR, ACEITAR COM ALERTA ou REJEITAR, a partir dos metadados do vídeo."""
    result = {"status": "ACEITAR", "resolucao": None, "fps": None, "motivos_rejeicao": [], "alertas": []}
    path = Path(path)
    if not path.is_file():
        result["motivos_rejeicao"].append(f"Vídeo não encontrado: {path}")
    else:
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 0.0
                result["resolucao"] = f"{stream.width}x{stream.height}"
                result["fps"] = round(fps, 2)
        except (av.error.FFmpegError, IndexError):
            result["motivos_rejeicao"].append(f"Não foi possível abrir um fluxo de vídeo em {path}.")
        else:
            # A precisão temporal de take_off e landing depende diretamente da taxa de quadros.
            if not fps:
                result["alertas"].append("O vídeo não informa a taxa de quadros.")
            elif fps < RECOMMENDED_FPS:
                result["alertas"].append(
                    f"FPS abaixo do recomendado ({RECOMMENDED_FPS}): {fps:.2f}. "
                    "O intervalo de incerteza dos eventos tende a ser maior.")
    if result["motivos_rejeicao"]:
        result["status"] = "REJEITAR"
    elif result["alertas"]:
        result["status"] = "ACEITAR COM ALERTA"
    return result
