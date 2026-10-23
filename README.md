# SpikeAI: saída do chão e aterrissagem

Um clipe, um atacante e dois eventos: `take_off` (último apoio antes do voo) e `landing`
(primeiro apoio depois do voo). O RTMPose-m HALPE26 fornece os pés; o
[Footskate Reducer](https://github.com/vt-vl-lab/footskate_reducer) fornece as
probabilidades de contato, com os pesos originais e sem novo treinamento. Uma referência
local 2D do piso, ajustada a partir dos pés com contato previsto, confere esses contatos.

## Executar

Extraia em uma pasta nova. Testado em Python 3.12; requer Python 3.10 ou superior.

```bash
bash instalar.sh
source .venv/bin/activate
python src/main.py input/ataque_completo.mp4 --selecionar
```

Marque o atacante na imagem inicial e pressione Enter. Os frames dos eventos
são calculados automaticamente. Modelos são baixados na primeira execução.

Sem interface gráfica, execute sem `--selecionar`, veja `athletes.jpg` gerado
na pasta de saída e repita com `--atleta N`. O índice escolhe a pessoa no primeiro
frame; o rastreador associa essa pessoa nos demais frames. Em cruzamentos ou
oclusões prolongadas, a associação pode falhar e precisa ser conferida no vídeo.

Opções: `--saida PASTA` (padrão `output/NOME_DO_VIDEO`; use uma pasta nova a cada execução)
e `--limiar` (limiar de contato do Footskate, padrão 0.5).

## Estrutura

```
src/
  main.py             orquestra: validar → pose → contato → piso → eventos → exportar
  validator.py        validação do vídeo de entrada (ACEITAR / ACEITAR COM ALERTA / REJEITAR)
  attacker.py         escolha inicial e rastreio do atacante
  footskate.py        HALPE26 → BODY25 e probabilidades de contato do Footskate Reducer
  ground.py           referência local 2D do piso e conferência dos contatos
  algorithm_rules.py  EVENTS e detect_events: estados de contato → take_off / landing
  export_handler.py   events.json, events.csv, contacts.csv e inspection.mp4
tests/                regras, pose/identidade, piso, validação e exportação
input/  output/       clipes de entrada e resultados (um subdiretório por vídeo)
```

## Como os eventos são decididos

Cada frame recebe um estado: `1` apoio, `0` sem apoio, `-1` evidência insuficiente.

- O Footskate dá a probabilidade de contato de dedo e calcanhar de cada pé. O movimento do
  fundo é compensado e a referência do piso ajuda a rejeitar apoio previsto no ar.
- Falhas curtas, de até `0.12 s`, dentro de um mesmo estado não dividem o salto.
- Transições com ambiguidade de até `0.15 s` retornam uma estimativa e um intervalo de frames,
  em vez de descartar o evento.
- Saída e aterrissagem são independentes. Se o clipe termina antes do pouso, a saída pode ser
  estimada e a aterrissagem continua inconclusiva.
- Sem referência de piso utilizável, o programa usa apenas o Footskate e registra isso em
  `source` (`footskate` em vez de `footskate+ground_reference`).

A referência é uma **aproximação local na imagem**, não uma calibração física do plano da
quadra nem uma segmentação visual do chão. Pode falhar com movimento em profundidade, câmera
que não possa ser estabilizada ou contatos previstos incorretos.

## Arquivos gerados

Na pasta `output/NOME_DO_VIDEO/`:

- `inspection.mp4`: esqueleto, referência do piso e marcações dos dois eventos. A linha
  amarela é a referência; pontos verdes indicam apoio estimado, vermelhos ausência de apoio
  e cinza evidência insuficiente. Não tem áudio.
- `events.json` e `events.csv`: status, frame, tempo, intervalo e origem de cada evento.
- `contacts.csv`: probabilidades de contato e estados por frame, antes (`state_evidence`) e
  depois (`state`) do tratamento temporal, para conferir sem repetir o RTMPose.

`events.json` traz `video_id`, `pipeline_status` e `alertas`, além dos eventos:

- `pipeline_status`: `APROVADO` se os dois eventos têm frame; `PENDENTE_REVISAO_MANUAL` se
  algum ficou inconclusivo.
- `alertas`: avisos do vídeo de entrada (por exemplo, FPS abaixo de 60) e o motivo de cada
  evento inconclusivo. Só as pendências alteram o `pipeline_status`.

Status de cada evento: `detected` é uma transição direta do modelo; `estimated` é uma
transição com intervalo curto ou resultado conferido pelo piso; `inconclusive` mantém
frame e tempo nulos. O intervalo descreve a ambiguidade temporal observada; **não é uma
margem de erro estatística nem garante conter o frame verdadeiro**. Precisão ainda precisa
de validação com anotações independentes.

Frames começam em 0. Tempos começam no primeiro frame e usam os timestamps reais (PTS) do
vídeo.

## Valores de referência

Resultados obtidos nos clipes de `input/` antes desta limpeza, para conferir uma nova execução.
As regras de decisão não mudaram. Não há anotações de verdade para medir precisão.

| Clipe | Saída estimada | Aterrissagem estimada |
| --- | --- | --- |
| ataque_braço | 52 (1.781667 s), intervalo 52–53 | 78 (2.640000 s), intervalo 77–79 |
| ataque_takeoff | 114 (3.776667 s) | Inconclusiva: clipe termina antes do pouso |
| ataque_completo | 77 (2.552333 s), intervalo 76–78 | 96 (3.185667 s), intervalo 95–97 |

## Testes

```bash
python -m unittest discover -s tests
```

Arquitetura, janela de 7 frames, normalização e filtro do Footskate vêm do commit