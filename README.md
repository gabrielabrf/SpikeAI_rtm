# SpikeAI: saída do chão e pouso

Um clipe, um atacante e dois eventos. O RTMPose HALPE26 fornece os pontos;
o Footskate Reducer estima contato. A referência local do chão confere os apoios,
e a sequência apoio → voo → apoio fornece saída e pouso.

## Executar

Python 3.10 ou superior; verificado em Python 3.12 com CPU.

```bash
bash instalar.sh
source .venv/bin/activate
python src/main.py input/ataque_completo.mp4 --selecionar
```

Marque o atacante e pressione Enter. Sem interface gráfica, execute sem
`--selecionar`, confira `athletes.jpg` gerado na saída e repita com `--atleta N`.
O atacante é escolhido no primeiro frame; os frames dos eventos são automáticos.
Os modelos são baixados na primeira execução.

Opções existentes: `--saida PASTA` e `--limiar` (contato; padrão 0.5).
Use uma pasta de saída nova para cada execução.

## Saída

- `inspection.mp4`: esqueleto, linha amarela e destaque nos eventos; sem áudio.
- `events.json` e `events.csv`: nome, status, frame, tempo e intervalo de cada evento.
- `contacts.csv`: probabilidades e estados por frame.

Frames começam em zero. Os tempos usam os timestamps reais do vídeo.
`take_off` é o último apoio antes do voo, distinto do ápice; `landing` é o primeiro
apoio depois do voo. Falhas de até 0,15 s entre observações do mesmo estado não
partem o salto. Transições imprecisas retornam intervalo ou `inconclusive`, com
frame e tempo nulos. Intervalos não são garantias de erro máximo.

A linha e a decisão de contato usam a mesma referência local. Apoios estáveis
atualizam o chão com limites proporcionais ao corpo e compensação da câmera.
Uma perda de câmera não atualiza o registro; uma associação de identidade
incompatível não fornece pontos do jogador vizinho.

É uma aproximação 2D. Oclusões, enquadramento e erros dos modelos podem impedir
a localização do evento. `pipeline_status=APROVADO` indica que ambos têm frame;
não representa validação humana nem medição de precisão.

