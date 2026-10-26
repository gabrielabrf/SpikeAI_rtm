import os
import yaml

def load_athlete_config(candidate_name: str, config_dir: str = "config/candidatos") -> dict:
    """
    Carrega as configurações específicas do atleta a partir do seu arquivo YML.
    Example candidate_name: 'candidato_a'
    """
    # Garante a extensão .yml
    if not candidate_name.endswith('.yml') and not candidate_name.endswith('.yaml'):
        filename = f"{candidate_name}.yml"
    else:
        filename = candidate_name

    file_path = os.path.join(config_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de configuração do atleta não encontrado em: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data


def select_athlete(candidate_id: str):
    """
    Realiza a seleção do atleta e retorna os parâmetros de inicialização do tracking.
    """
    athlete_data = load_athlete_config(candidate_id)
    
    print(f"[INFO] Atleta selecionado: {athlete_data.get('nome', candidate_id)}")
    
    # Extrai os dados principais como bbox inicial ou id do rastreio
    tracking_info = {
        "id": athlete_data.get("id"),
        "nome": athlete_data.get("nome"),
        "initial_bbox": athlete_data.get("initial_bbox", None),
        "kp_ref": athlete_data.get("keypoints_ref", None)
    }
    
    return tracking_info