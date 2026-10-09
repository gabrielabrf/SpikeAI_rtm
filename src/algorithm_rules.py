import numpy as np

class SpikeAIRuleEngine:
    def __init__(self, fps=60):
        self.fps = fps

    def detectar_inicio_backswing(self, wrist_x_trajectory, threshold_movimento=2.0):
        """
        Regra Operacional: Primeiro frame em que a mão do braço de ataque 
        inicia deslocamento consistente para trás (inversão do vetor de velocidade no eixo X/Y).
        """
        velocidades = np.diff(wrist_x_trajectory)
        # Identifica onde a mão começa a se mover para trás de forma sustentada
        for i, vel in enumerate(velocidades):
            if vel < -threshold_movimento:
                return i  # Índice do frame correspondente ao início do backswing
        return None

    def detectar_cocking(self, shoulder_wrist_distances):
        """
        Regra Operacional: Frame em que o braço atinge sua extensão máxima 
        de armação (ponto de inflexão da distância ou rotação máxima do ombro) antes da aceleração.
        """
        if not shoulder_wrist_distances:
            return None
        # O pico de extensão/distância máxima antes do golpe
        peak_frame = int(np.argmax(shoulder_wrist_distances))
        return peak_frame

    def detectar_take_off(self, ankle_y_trajectory, ground_threshold=5.0):
        """
        Regra Operacional: Último frame com contato visível do atleta com o solo 
        antes da fase aérea (início da subida abrupta no eixo Y dos tornozelos/pés).
        """
        velocidades_y = np.diff(ankle_y_trajectory)
        # O take-off ocorre logo antes do tornozelo se mover rapidamente para cima (subida na tela)
        for i in range(len(velocidades_y) - 1, -1, -1):
            if velocidades_y[i] < -ground_threshold: # Movimento ascendente na imagem
                # Retorna o último frame de contato estável
                return max(0, i - 1)
        return None

    def detectar_landing(self, ankle_y_trajectory, ground_level):
        """
        Regra Operacional: Primeiro frame com contato visível de qualquer parte 
        do pé com o solo após a fase aérea (retorno ao nível do solo y).
        """
        for i, y_pos in enumerate(ankle_y_trajectory):
            # Se o pé retorna à altura do solo após ter saído
            if abs(y_pos - ground_level) < 3.0:
                return i
        return None

# Exemplo de aplicação prática do motor de regras
if __name__ == "__main__":
    engine = SpikeAIRuleEngine(fps=60)
    
    # Dados simulados de trajetórias articulares extraídas de um vídeo
    trajetoria_punho_x = [100, 102, 105, 103, 98, 90, 80] # Exemplo de recuo (backswing)
    distancias_ombro_punho = [40, 45, 55, 65, 78, 70, 50]   # Pico de cocking no índice 4
    
    frame_backswing = engine.detectar_inicio_backswing(trajetoria_punho_x, threshold_movimento=1.5)
    frame_cocking = engine.detectar_cocking(distancias_ombro_punho)
    
    print("--- Resultado da Execução da Task 3 (Regras Algorítmicas) ---")
    print(f"Frame detectado - Início do Backswing: {frame_backswing}")
    print(f"Frame detectado - Cocking Máximo: {frame_cocking}")
    