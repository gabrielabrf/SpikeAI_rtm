# Spike.AI - Motion Engine (Task 3 da Issue 6)

Pipeline de Pose Estimation com RTMPose (via `rtmlib`) para análise biomecânica do ataque no voleibol. O sistema calcula métricas temporais/espaciais 2D, identifica momentos críticos do gesto motor e exporta os dados estruturados.

---

## 1. Matriz de Requisitos Computacionais e Biomecânicos

A tabela a seguir consolida a conversão dos casos de uso em requisitos computacionais explícitos:

| Categoria | Requisito Registrado / Especificação |
| :--- | :--- |
| **Landmarks Obrigatórios** | Nariz (referência de altura), Ombros, Cotovelos, Punhos, Quadris, Joelhos, Tornozelos, Calcanhares e Ponta dos pés. |
| **Features Derivadas** | • **Ângulos Articulares Projetados:** Cotovelo, Ombro, Quadril e Joelho.<br>• **Velocidade Aparente:** Velocidade escalar do punho e velocidade angular aparente das articulações.<br>• **Cinemática Temporal:** Trajetória dos keypoints, duração total do salto, duração das fases e diferença temporal entre eventos (ex: tempo entre `cocking` e `peak_hand_velocity`). |
| **Eventos Mapeados** | `plant`, `take_off`, `jump_peak`, `cocking_backswing`, `peak_hand_velocity`, `landing`. |
| **Unidades de Medida** | Ângulos em Graus ($\circ$), Velocidades em Pixels/segundo ($\text{px/s}$), Velocidade Angular em $\circ/\text{s}$, Tempo em Segundos ($\text{s}$) e Altura em Pixels ($\text{px}$). |
| **Sistema de Coordenadas** | Matriz de Imagem 2D (Origem $(0,0)$ no canto superior esquerdo; $X$ cresce para a direita, $Y$ cresce para baixo). |
| **Confiança Mínima** | $\text{Threshold} \ge 0.5$ (`KPT_CONF_THRESHOLD`). |
| **Frequência Temporal** | Mínimo $30\text{ FPS}$ (Recomendado $60\text{ FPS}$ ou superior para mitigar borrão de movimento no ataque). |
| **Normalização Espacial** | `body_height_px` (distância vertical entre o Nariz e o ponto médio dos Tornozelos) usada para escala relativa. |
| **Dados Persistidos** | • **Série Temporal Bruta/Derivada:** Arquivo `.csv` (separador `;`).<br>• **Relatório de Auditoria:** Arquivo `.txt` formatado (tabela monospaced).<br>• **Mídia de Saída:** Vídeo `.mp4` anotado com a marcação de eventos. |
| **Limitações da Mediçâo 2D** | Impossibilidade de medir a rotação axial volumétrica anatômica real (ex: "rotação externa aparente do ombro em vídeo 2D" em vez de "rotação real do úmero"), oclusão de membros e distorção de perspectiva fora do plano focal. |

---

## 2. Eventos Biomecânicos Detectados

1. **Preparação e Impulso (`plant` / `take_off`)**: Início da desaceleração/flexão pré-salto e momento em que os pés perdem contato com o solo.
2. **Pico do Salto (`jump_peak`)**: Ápice da fase aérea (menor coordenada $Y$ dos tornozelos).
3. **Armação do Golpe (`cocking_backswing`)**: Ângulo máximo de extensão projetada do cotovelo/ombro antes da aceleração.
4. **Pico de Velocidade (`peak_hand_velocity`)**: Ponto de máxima velocidade escalar aparente do punho no tempo ($\text{px/s}$).
5. **Aterrissagem (`landing`)**: Retorno do contato dos pés com a altura base do solo.
6. **Diferença Temporal ($\Delta t$)**: Intervalo de tempo transcorrido entre a armação do golpe e o pico de velocidade da mão.

---

## 3. Especificação de Nomenclatura 2D (Isenção Técnica)

Para evitar erros conceituais no relatório técnico:
* **`*_angle_2d_proj`**: Indica o ângulo plano formado na projeção 2D da câmera.
* **`*_apparent_velocity_px_s`**: Refere-se à velocidade escalar aparente no plano focal medida em pixels por segundo.