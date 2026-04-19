# PLAN.md

> Arquivo vivo. Uma feature ou tarefa por vez. Ao concluir, arquive em
> `docs/plans/<data>-<slug>.md` e limpe este arquivo para a proxima.
>
> **Regra:** o agente atualiza este arquivo antes de codar e marca progresso a
> cada passo. Se o plano nao existe ou esta desatualizado, a implementacao para.

---

## 0. Metadados

- **Feature / Tarefa:** Encerramento - treino completo + IA modelo no browser
- **Slug:** `encerramento-treino-e-inferencia-browser`
- **Autor humano:** _a definir_
- **Data de inicio:** 2026-04-17
- **Branch:** `feat/encerramento-treino-inferencia`
- **Status:** Arquivado e validado parcialmente

**Execucao atual: treino otimizado de 2.000 episodios a partir do melhor checkpoint (2026-04-18)**

- [x] **TRAIN.run2000.1** Adicionar uma progressao simples no treino para
  acompanhar quantos episodios ja rodaram e quantos faltam.
- [x] **TRAIN.run2000.2** Medir o score atual do `best.pt` que servira como
  ponto de partida.
- [x] **TRAIN.run2000.3** Rodar 2.000 episodios otimizados em cima dos
  melhores pesos atuais.
- [x] **TRAIN.run2000.4** Comparar score inicial e final, preservando o
  melhor resultado util para o projeto.
- [x] **TRAIN.run2000.5** Gerar um grafico simples da evolucao do treino.

**Resultado do treino otimizado de 2.000 episodios (2026-04-18)**

- [x] Ponto de partida: `training/checkpoints/clean-10000/best.pt`
  (episodio `1599`).
- [x] Score inicial medido com a mesma avaliacao de 100 episodios:
  `eval_hit_rate=0.5185` (`126` hits, `117` misses).
- [x] Treino executado com
  `uv run python training/scripts/train.py --episodes 2000 --seed 42 --device cuda --batch-envs 8 --lr 3e-4 --hit-reward 1.0 --miss-penalty -1.5 --resume-from training/checkpoints/clean-10000/best.pt --checkpoint-dir training/checkpoints/tune-from-best-2000 --checkpoint-every 50 --debug-snapshot-count 0 --progress-interval 50 --evaluation-interval 100 --evaluation-episodes 30`.
- [x] Progresso simples emitido durante a execucao em
  `training-run-2000.log`, no formato `progress X de 2000 (faltam Y)`.
- [x] Fim da rodada: `2.000 de 2.000` episodios adicionais concluidos.
- [x] Melhor checkpoint da rodada: `training/checkpoints/tune-from-best-2000/best.pt`
  no episodio absoluto `1699`, com `eval_hit_rate=0.529` na avaliacao curta
  da propria rodada.
- [x] Comparacao justa com avaliacao fixa de 100 episodios:
  novo `best.pt` da rodada ficou em `eval_hit_rate=0.4979`
  (`119` hits, `120` misses), pior que o score inicial `0.5185`.
- [x] `latest.pt` da rodada terminou ainda pior, com `eval_hit_rate=0.4000`
  (`102` hits, `153` misses).
- [x] Decisao de engenharia: **nao** substituir o modelo atual publicado no
  jogo, porque a nova rodada nao superou o baseline medido com a mesma regua.
- [x] Grafico da evolucao gerado em
  `training/reports/tune-from-best-2000-evolution.png`.

**Execucao atual: reduzir ruido da avaliacao e endurecer selecao do melhor modelo (2026-04-18)**

- [x] **TRAIN.robust.1** Tornar a avaliacao menos ruidosa para selecao de
  checkpoint, usando uma metrica mais conservadora do que o `hit_rate` bruto.
- [x] **TRAIN.robust.2** Atualizar a selecao de `best.pt` para usar esse
  criterio mais duro.
- [x] **TRAIN.robust.3** Cobrir a nova logica com testes automatizados e
  revalidar a suite relevante.
- [x] **TRAIN.robust.4** Rodar mais `2.000` episodios a partir do melhor
  checkpoint atual com a nova estrategia.
- [x] **TRAIN.robust.5** Comparar score inicial e final da nova rodada.

**Resultado da rodada robusta de 2.000 episodios (2026-04-18)**

- [x] Avaliacao atualizada para calcular `attempt_count` e usar
  `hit_rate_lower_bound` (limite inferior de Wilson 95%) como metrica de
  selecao do `best.pt`, mantendo `hit_rate` bruto apenas como apoio.
- [x] Retomada de checkpoint ficou retrocompativel: historicos antigos sem
  `attempt_count` ou `hit_rate_lower_bound` passam a ser reidratados com os
  valores derivados corretamente.
- [x] CLI de treino passou a reportar tanto `eval_hit_rate` quanto
  `eval_hit_rate_lb`, e os hparams agora registram
  `evaluation_selection_metric=hit_rate_wilson_lower_bound_95`.
- [x] Suite validada apos a mudanca:
  `uv run ruff check ...`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [x] Score base reavaliado em
  `training/checkpoints/clean-10000/best.pt` (episodio `1599`):
  `hit_rate=0.518519`, `hit_rate_lower_bound=0.455897`,
  `126` hits, `117` misses.
- [x] Nova rodada executada com
  `uv run python training/scripts/train.py --episodes 2000 --seed 42 --device cuda --batch-envs 8 --lr 3e-4 --hit-reward 1.0 --miss-penalty -1.5 --resume-from training/checkpoints/clean-10000/best.pt --checkpoint-dir training/checkpoints/tune-from-best-2000-robust --checkpoint-every 50 --debug-snapshot-count 0 --progress-interval 50 --evaluation-interval 100 --evaluation-episodes 100`.
- [x] Melhor checkpoint interno da rodada ficou em
  `training/checkpoints/tune-from-best-2000-robust/best.pt`
  (episodio absoluto `3099`) com avaliacao da propria rodada em
  `eval_hit_rate=0.502092` e `eval_hit_rate_lb=0.439172`.
- [x] Comparacao justa com a mesma avaliacao fixa de 100 episodios:
  `robust_round_best` ficou em `hit_rate=0.516667`,
  `hit_rate_lower_bound=0.453680` (`124` hits, `116` misses), ainda
  ligeiramente abaixo do baseline.
- [x] `latest.pt` da rodada degradou para `hit_rate=0.400000`,
  `hit_rate_lower_bound=0.341784` (`102` hits, `153` misses).
- [x] Decisao de engenharia: manter o modelo publicado no jogo apontando para
  o baseline `training/checkpoints/clean-10000/best.pt`; a rodada robusta
  aproximou o resultado, mas ainda nao o superou.
- [x] Grafico da rodada gerado em
  `training/reports/tune-from-best-2000-robust-evolution.png`.

**Execucao atual: inducao de retorno ao centro no reward shaping (2026-04-18)**

- [x] **TRAIN.center.1** Ajustar o reward shaping para incentivar a raquete
  direita a voltar ao centro e evitar movimento ocioso quando a bola ainda
  nao esta ameacando a defesa.
- [x] **TRAIN.center.2** Manter equivalencia entre treino unitario e treino
  batelado, evitando drift entre `reinforce.py` e `batched_reinforce.py`.
- [x] **TRAIN.center.3** Cobrir a nova logica com testes automatizados de
  recompensa e revalidar as suites afetadas.
- [x] **TRAIN.center.4** Rodar uma validacao real de `2.000` episodios a
  partir do melhor checkpoint atual com o shaping novo.
- [x] **TRAIN.center.5** Comparar a rodada nova com o baseline atual tanto
  em score quanto em metricas de postura/retorno ao centro.

**Resultado da rodada com inducao de retorno ao centro (2026-04-18)**

- [x] Reward shaping atualizado para combinar:
  recompensa de rebatida/falha, incentivo a reduzir a distancia ao centro,
  bonus pequeno por sustentar a raquete perto do centro em estados de espera
  e penalidade leve por movimento ocioso nesses estados.
- [x] Treino unitario e batelado foram mantidos equivalentes usando a mesma
  formula de shaping em `reward_shaping.py` e no caminho de batch.
- [x] Novos hiperparametros ficaram expostos na CLI e persistidos em
  checkpoint: `centering_reward_scale`, `center_hold_bonus`,
  `idle_movement_penalty` e `centering_window_ratio`.
- [x] Suites verdes apos a mudanca:
  `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [x] Baseline medido em `training/checkpoints/clean-10000/best.pt`:
  `hit_rate=0.518519`, `hit_rate_lower_bound=0.455897`,
  `waiting_center_distance_mean=0.697558`,
  `waiting_centered_rate=0.090601`,
  `waiting_move_rate=0.564084`.
- [x] Nova rodada executada com
  `uv run python training/scripts/train.py --episodes 2000 --seed 42 --device cuda --batch-envs 8 --lr 3e-4 --hit-reward 1.0 --miss-penalty -1.5 --centering-reward-scale 0.02 --center-hold-bonus 0.0005 --idle-movement-penalty 0.0005 --centering-window-ratio 0.65 --resume-from training/checkpoints/clean-10000/best.pt --checkpoint-dir training/checkpoints/tune-from-best-2000-centering --checkpoint-every 50 --debug-snapshot-count 0 --progress-interval 50 --evaluation-interval 100 --evaluation-episodes 100`.
- [x] Melhor checkpoint da rodada:
  `training/checkpoints/tune-from-best-2000-centering/best.pt`
  no episodio absoluto `3199`, com avaliacao interna de
  `eval_hit_rate=0.578475` e `eval_hit_rate_lb=0.512871`.
- [x] Comparacao justa na mesma avaliacao fixa de 100 episodios:
  `centering_round_best` superou o baseline em score, com
  `hit_rate=0.530364` e `hit_rate_lower_bound=0.468135`
  (`131` hits, `116` misses).
- [x] Comparacao de postura em estados de espera:
  `waiting_center_distance_mean` caiu de `0.697558` para `0.649919` e
  `waiting_centered_rate` subiu de `0.090601` para `0.102558`.
- [x] Sinal restante de ajuste fino:
  `waiting_move_rate` piorou de `0.564084` para `0.659223`, indicando que a
  centralizacao media melhorou, mas ainda com excesso de movimento ocioso.
- [x] `latest.pt` da rodada nao foi bom candidato final
  (`hit_rate=0.381132`, `hit_rate_lower_bound=0.324750`), entao a publicacao
  usou explicitamente o `best.pt`.
- [x] Novo ONNX publicado com
  `uv run python training/scripts/export.py --checkpoint training/checkpoints/tune-from-best-2000-centering/best.pt --output web/public/model.onnx`,
  com paridade PyTorch <-> ONNX verde (`max_abs_diff=5.722e-06`).
- [x] Grafico da rodada gerado em
  `training/reports/tune-from-best-2000-centering-evolution.png`.

**Execucao atual: estabilizacao do treino do modelo (2026-04-18)**

- [x] **TRAIN.stable.1** Salvar `best.pt` imediatamente quando a metrica de
  selecao melhorar, sem depender apenas da cadencia de checkpoint.
- [x] **TRAIN.stable.2** Adicionar avaliacao separada do treino baseada em
  taxa de rebatida do paddle direito.
- [x] **TRAIN.stable.3** Implementar early stopping guiado por avaliacao para
  interromper rodadas que estejam degradando.
- [x] **TRAIN.stable.4** Expor a nova configuracao na CLI de treino e
  persistir o historico de avaliacao nos checkpoints.
- [x] **TRAIN.stable.5** Cobrir com testes automatizados e revalidar a suite
  completa.

**Resultado da estabilizacao do treino (2026-04-18)**

- [x] `best.pt` agora e salvo imediatamente quando a metrica de selecao
  melhora, sem depender apenas de `checkpoint_every`.
- [x] Avaliacao separada do treino implementada com taxa de rebatida
  (`hit_rate`) do paddle direito.
- [x] Early stopping implementado e guiado por avaliacao.
- [x] Historico de avaliacao persistido em checkpoint para analise e
  retomada mais informada.
- [x] CLI de treino atualizada com:
  `--evaluation-interval`, `--evaluation-episodes`,
  `--early-stopping-patience`, `--early-stopping-min-improvement`.
- [x] Suite completa verde:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [x] Smoke real da CLI validado com early stopping:
  `uv run python training/scripts/train.py --episodes 5 --seed 42 --device cpu --batch-envs 1 --checkpoint-dir training/checkpoints/smoke-stable-train --checkpoint-every 5 --debug-snapshot-count 0 --evaluation-interval 1 --evaluation-episodes 2 --early-stopping-patience 2 --early-stopping-min-improvement 0.0`,
  resultando em parada antecipada com `episodes=4` e
  `eval_hit_rate=0.250`.

**Execucao atual: limpeza do HUD e ajuste responsivo do frontend (2026-04-18)**

- [x] **UI.clean.1** Remover completamente o HUD de debug dos modelos no
  frontend.
- [x] **UI.clean.2** Ajustar o layout para caber no viewport do browser,
  mantendo score, status, controles e jogo visiveis de forma organizada.
- [x] **UI.clean.3** Colocar `Controle esquerda` e `Controle direita` na
  mesma linha, cada um centralizado em sua metade.
- [x] **UI.clean.4** Atualizar testes web para refletir a UI sem HUD e a nova
  composicao responsiva.
- [x] **UI.clean.5** Revalidar checks e manter o servidor local pronto para
  avaliacao.

**Resultado da limpeza do HUD e ajuste responsivo (2026-04-18)**

- [x] HUD de debug removido do frontend; a tela agora mostra apenas score,
  status, controles e o jogo.
- [x] Layout ajustado para ocupar o viewport do browser de forma mais
  previsivel, com o canvas reduzindo para caber em janelas menores.
- [x] `Controle esquerda` e `Controle direita` ficaram na mesma linha, cada
  um centralizado em sua metade.
- [x] Testes web atualizados para a UI sem HUD e para a nova composicao.
- [x] Validacoes verdes:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [x] Frontend local segue ativo em `http://localhost:5173/`.

**Execucao atual: controle configuravel dos dois paddles (2026-04-18)**

- [x] **CTRL.dual.1** Adicionar um seletor para a raquete esquerda com as
  mesmas opcoes da direita: `modelo`, `heuristica`, `teclado`.
- [x] **CTRL.dual.2** Separar o teclado por lado: esquerda em `W/S`,
  direita em `ArrowUp/ArrowDown`.
- [x] **CTRL.dual.3** Permitir `modelo vs modelo`, reaproveitando o mesmo
  ONNX e espelhando horizontalmente o bitmap da esquerda antes da inferencia.
- [x] **CTRL.dual.4** Atualizar HUD/status para refletir os modos dos dois
  lados e expor debug do modelo por lado quando aplicavel.
- [x] **CTRL.dual.5** Cobrir com testes automatizados web e revalidar a
  execucao local do frontend.

**Resultado do controle configuravel dos dois paddles (2026-04-18)**

- [x] UI atualizada com dois seletores independentes:
  `Controle esquerda` e `Controle direita`, cada um com
  `modelo | heuristica | teclado`.
- [x] Teclado separado por lado:
  esquerda em `W/S`, direita em `ArrowUp/ArrowDown`.
- [x] Runtime do browser atualizado para suportar `modelo vs modelo`
  com dois `FrameStack`s independentes e o mesmo `model.onnx`.
- [x] Observacao da esquerda convertida para a perspectiva canonica do
  treino por espelhamento horizontal do bitmap antes da inferencia.
- [x] HUD/status atualizado para mostrar o modo atual de cada lado e
  debug do modelo por lado quando aplicavel.
- [x] Testes/validacoes verdes:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [x] Frontend local validado em `http://localhost:5173/` com
  `GET / -> 200` e `HEAD /model.onnx -> 200`.

**Execucao atual: treino limpo otimizado de 10.000 episodios (2026-04-18)**

- [x] **TRAIN.clean10000.1** Rodar o treino limpo otimizado com `10.000`
  episodios, `device=cuda`, `batch_envs=8` e apenas `5` snapshots de debug.
- [x] **TRAIN.clean10000.2** Exportar o melhor checkpoint limpo para
  `web/public/model.onnx`, com fallback explicito para `latest.pt` se
  necessario.
- [x] **TRAIN.clean10000.3** Converter os bitmaps de debug gerados para
  `.png`, preservando apenas o conjunto enxuto desta rodada.
- [x] **TRAIN.clean10000.4** Reexecutar as verificacoes relevantes de Python,
  paridade e web apos a exportacao.
- [x] **TRAIN.clean10000.5** Subir o frontend com o modelo novo servido e
  registrar a URL pronta para teste manual do jogo.

**Resultado da execucao limpa de 10.000 episodios (2026-04-18)**

- [x] Treino executado com
  `uv run python training/scripts/train.py --episodes 10000 --seed 42 --device cuda --batch-envs 8 --checkpoint-dir training/checkpoints/clean-10000 --checkpoint-every 100 --debug-output-dir training/debug/bitmaps`.
- [x] Tempo total observado do treino: `9868.3s` (aprox. `2h 44min 28s`).
- [x] Resultado final do treino: `episodes=10000`, `final_loss=0.0000`,
  `final_reward=-3.00`, `avg_reward_window=-0.94`, `device=cuda`,
  `batch_envs=8`.
- [x] Checkpoints limpos gerados em
  `training/checkpoints/clean-10000/latest.pt` e
  `training/checkpoints/clean-10000/best.pt`.
- [x] Exportacao ONNX publicada com
  `uv run python training/scripts/export.py --checkpoint training/checkpoints/clean-10000/best.pt --output web/public/model.onnx`.
- [x] Paridade PyTorch <-> ONNX validada em 50 amostras:
  `max_abs_diff=5.960e-07`, dentro da tolerancia `1e-4`.
- [x] `web/public/model.onnx` atualizado com `19,444,909` bytes
  (aprox. `18.54 MiB`).
- [x] Apenas 5 snapshots de debug mantidos em `training/debug/bitmaps/`,
  com conversao local para `.png`.
- [x] Verificacoes automatizadas verdes:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [x] Frontend local ativo em `http://localhost:5173/`, com
  `GET / -> 200` e `HEAD /model.onnx -> 200`.

**Debug atual: exemplos de bitmap em diretório dedicado (2026-04-18)**

- [x] **DEBUG.bitmap.1** Persistir exemplos da observacao do modelo em um
  diretório apropriado do projeto, sem espalhar arquivos no repositório.
- [x] **DEBUG.bitmap.2** Manter apenas os 10 bitmaps mais recentes em buffer
  rotativo no disco.
- [x] **DEBUG.bitmap.3** Validar com testes automatizados e gerar exemplos
  reais para inspecao manual.

**Debug atual: conversao de `.pgm` para `.png` no Windows (2026-04-18)**

- [x] **DEBUG.png.1** Criar um conversor local de `.pgm` para `.png` sem
  depender de bibliotecas externas novas.
- [x] **DEBUG.png.2** Expor um script para converter todos os bitmaps de
  debug da pasta apropriada de uma vez.
- [x] **DEBUG.png.3** Validar com testes e gerar os `.png` reais para uso
  imediato.

**Ajuste atual: reduzir reflexo da IA estocastica (2026-04-18)**

- [x] **TUNE.reaction.1** Dobrar o tempo de reacao da IA estocastica,
  aumentando `ai.inference_interval_ticks` de `4` para `8` na configuracao
  compartilhada.
- [x] **TUNE.reaction.2** Regenerar configs derivadas e validar rapidamente
  que web/paridade seguem consistentes.
- [x] **TUNE.reaction.3** Subir o servidor web para avaliacao manual.

**Correcao atual: restaurar model e atrasar heuristica (2026-04-18)**

- [x] **TUNE.heuristic.1** Restaurar `ai.inference_interval_ticks` para `4`
  e adicionar uma configuracao separada para a frequencia de decisao da
  heuristica.
- [x] **TUNE.heuristic.2** Fazer paddles controlados por heuristica decidirem
  apenas a cada `2` ticks, reutilizando a ultima acao entre decisoes.
- [x] **TUNE.heuristic.3** Regenerar configs, validar web/paridade e manter o
  servidor no ar para avaliacao.

**Ajuste atual: alinhar heuristica ao modelo (2026-04-18)**

- [x] **TUNE.heuristic.match1** Ajustar `ai.heuristic_interval_ticks` de `2`
  para `4`, igualando a heuristica ao tempo de reacao do modelo.
- [x] **TUNE.heuristic.match2** Regenerar configs derivadas e validar
  consistencia rapida de web/paridade.

**Ajuste atual: desacelerar heuristica para avaliacao (2026-04-18)**

- [x] **TUNE.heuristic.slow1** Ajustar `ai.heuristic_interval_ticks` de `4`
  para `12`, deixando a heuristica responder a cada 12 ticks.
- [x] **TUNE.heuristic.slow2** Regenerar configs derivadas e validar
  consistencia de Python/web/paridade.

**Ajuste atual: heuristica com 1 segundo de reacao (2026-04-18)**

- [x] **TUNE.heuristic.1s1** Ajustar `ai.heuristic_interval_ticks` de `12`
  para `60`, deixando a heuristica responder a cada 1 segundo.
- [x] **TUNE.heuristic.1s2** Regenerar configs derivadas e validar
  consistencia de Python/web/paridade.

**Reinicio atual: novo pipeline de treino limpo (2026-04-18)**

- [x] **RESET.train.1** Reconfigurar a observacao temporal para 5 frames com
  stride fixo de 5 ticks, compartilhado entre treino e runtime web.
- [x] **RESET.train.2** Ajustar a cadencia do modelo para 30 ticks
  (~0,5 segundo) e manter a heuristica em 60 ticks, validando a sincronia
  2:1 entre modelo e heuristica.
- [x] **RESET.train.3** Trocar a recompensa para foco exclusivo em rebater:
  reforco positivo em rebatidas da direita e negativo em falhas da direita,
  sem incentivar placar como objetivo principal.
- [x] **RESET.train.4** Substituir o debug de bitmaps por apenas 5 snapshots
  por treino, capturados em memoria e materializados fora do hot loop.
- [x] **RESET.train.5** Criar testes automatizados de observacao temporal,
  sanidade do treino, benchmark curto de 10 episodios e integracao do modelo
  no browser com as novas regras.
- [x] **RESET.train.6** Apagar o modelo ONNX atual e os artefatos de treino
  anteriores, deixando o projeto pronto para um treino realmente limpo.

**Correcao atual em andamento (2026-04-17)**

- [x] **BUG.model-load.1** Reproduzir e isolar a falha de carregamento do
  modelo no browser quando o modo `model` e selecionado.
- [x] **BUG.model-load.2** Ajustar a inicializacao de `onnxruntime-web` para
  um caminho compativel com Vite e browser local, sem depender de fallback
  fragil de assets WASM/threading.
- [x] **BUG.model-load.3** Cobrir com teste de regressao no web e rodar
  `pnpm --filter web typecheck && pnpm --filter web lint && pnpm --filter web test`.

**Causa raiz identificada (2026-04-17)**

- [x] A `Content-Security-Policy` do `web/index.html` bloqueava a
  compilacao/instanciacao do WebAssembly do `onnxruntime-web`. O erro real em
  browser era: `no available backend found ... WebAssembly.instantiate() ...
  violates Content Security Policy`.
- [x] Correcao aplicada: `script-src 'self' 'wasm-unsafe-eval'` em
  `web/index.html`, mantendo a liberacao no menor escopo necessario para o
  runtime ONNX no browser.

**Continuacao de treino solicitada (2026-04-17)**

- [x] **TRAIN.plus100.1** Retomar o treino a partir de
  `training/checkpoints/final-100/latest.pt` por mais 100 episodios, sem
  reinicializar pesos.
- [x] **TRAIN.plus100.2** Exportar o checkpoint continuado resultante para
  `web/public/model.onnx`.
- [x] **TRAIN.plus100.3** Validar que o artefato novo esta sendo servido pelo
  frontend e que o modo `model` volta a subir sem fallback.

**Resultado da continuacao (2026-04-17)**

- [x] Treino retomado com
  `uv run python training/scripts/train.py --episodes 100 --reward-shaping --seed 42 --device cuda --batch-envs 8 --resume-from training/checkpoints/final-100/latest.pt --checkpoint-dir training/checkpoints/final-200-resumed`.
- [x] Saida do treino: `episodes=200`, `final_loss=-0.0590`,
  `final_reward=-2.00`, `avg_reward_window=-1.59`, `device=cuda`,
  `batch_envs=8`.
- [x] Checkpoint continuado gerado em
  `training/checkpoints/final-200-resumed/latest.pt`.
- [x] Nao houve `best.pt` novo nesse diretório; os 100 episodios adicionais nao
  superaram o melhor score historico registrado. Para publicar o modelo
  continuado no jogo, foi exportado o `latest.pt`.
- [x] Exportacao validada: `parity samples=50`, `max_abs_diff=2.980e-07`,
  dentro da tolerancia `1e-4`.
- [x] `web/public/model.onnx` atualizado em `2026-04-17 22:59:52`.
- [x] Frontend validado em `http://localhost:5173/`: status `Direita: modelo`
  sem fallback em execucao headless.

**HUD de debug do modelo (2026-04-17)**

- [x] **HUD.model.1** Exibir no frontend um HUD de debug com
  probabilidades `up/down/none`, `argmax` e ultima acao aplicada quando o
  modo da direita for `model`.
- [x] **HUD.model.2** Mostrar estado de espera antes da primeira inferencia,
  sem poluir os modos `heuristic` e `keyboard`.
- [x] **HUD.model.3** Validar com testes web que o HUD aparece com os dados
  esperados e permanece vazio fora do modo `model`.

**Execucao solicitada: D5 oficial com 10.000 episodios totais (2026-04-17)**

- [x] **D5.10000.a** Retomar a partir de
  `training/checkpoints/final-200-resumed/latest.pt` e completar o total de
  `10.000` episodios com treino otimizado (`cuda` + `batch_envs=8`).
- [x] **D6.10000.b** Exportar o modelo resultante para `web/public/model.onnx`,
  preferindo `best.pt` quando existir e usando `latest.pt` como fallback.
- [x] **D7.10000.c** Validar o frontend com o modelo novo servido em
  `http://localhost:5173/`.
- [x] **D8.10000.d** Revalidar criterios automatizados e registrar pendencias
  manuais restantes.
- [x] **D9.10000.e** Rodar a suite completa de Python, paridade e web.
- [x] **D10.10000.f** Arquivar o estado final do plano desta execucao em
  `docs/plans/`.

**Resultado do D5 oficial 10.000 (2026-04-18)**

- [x] Treino oficial executado com
  `uv run python training/scripts/train.py --episodes 9800 --reward-shaping --seed 42 --device cuda --batch-envs 8 --resume-from training/checkpoints/final-200-resumed/latest.pt --checkpoint-dir training/checkpoints/d5-10000-total`.
- [x] Resultado final da rodada oficial: `episodes=10000`,
  `final_loss=-0.0000`, `final_reward=-3.00`,
  `avg_reward_window=-1.61`, `device=cuda`, `batch_envs=8`.
- [x] Checkpoint oficial gerado em
  `training/checkpoints/d5-10000-total/latest.pt`.
- [x] Nao houve `best.pt` novo nessa rodada; o treino nao superou o melhor
  reward medio historico durante a execucao. Para publicacao web, foi usado o
  `latest.pt` como fallback explicito do D6.
- [x] Exportacao ONNX executada com
  `uv run python training/scripts/export.py --checkpoint training/checkpoints/d5-10000-total/latest.pt --output web/public/model.onnx`.
- [x] Paridade PyTorch <-> ONNX validada em 50 amostras:
  `max_abs_diff=7.629e-06`, dentro da tolerancia `1e-4`.
- [x] `web/public/model.onnx` atualizado em `2026-04-18 00:51:00`.
- [x] Frontend validado em `http://localhost:5173/`: `HTTP 200`, artefato
  `model.onnx` servindo e pagina inicializando em modo `Direita: modelo`.
- [x] Checks automatizados verdes:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [ ] Pendente: smoke visual/manual completo de uma partida `0 -> 21`,
  observando comportamento do modelo, HUD de debug e ausencia de erro no
  console em sessao real.

**Correcao: politica estocastica no browser (2026-04-18)**

- [x] **FIX.policy.1** Alinhar a inferencia do browser com o treino:
  amostrar a acao pelas probabilidades do modelo em vez de congelar no
  `argmax`.
- [x] **FIX.policy.2** Preservar no HUD a diferenca entre `argmax` e `ultima`
  acao aplicada.
- [x] **FIX.policy.3** Cobrir com teste web que prove que o browser pode agir
  diferente do `argmax` quando a distribuicao assim permitir.

---

## 1. Contexto e Problema

- **Problema:** a Fase 2 entregou a fundacao (frame_stack, rollout, model, policy
  e uma iteracao REINFORCE), mas falta o pipeline real de treino multi-episodio,
  exportacao ONNX, inferencia no browser e a substituicao do controle de teclado
  pela politica treinada na raquete direita. Sem isso, o objetivo final do
  projeto nao e atingido.
- **Quem sofre:** o usuario nao consegue ver o sistema funcionando ponta a
  ponta (heuristica estocastica vs modelo treinado).
- **Por que agora:** todas as dependencias estao no lugar (motor com paridade,
  modelo, policy, iteracao REINFORCE testada, ambiente PyTorch sincronizado).
  Falta apenas integrar e treinar.

---

## 2. Critica da Ideia

- **Faz sentido tecnico?** Sim. As pecas estao prontas e o caminho restante e
  integracao incremental.
- **Ja existe algo no codigo que resolve isso?** Nao. Pipeline real, exportacao
  e inferencia no browser sao todos ineditos.
- **Existe solucao mais simples?** Treinar inline em notebook e copiar o modelo
  manualmente. Rejeitada porque quebra reprodutibilidade e sai do padrao do
  repositorio.
- **Riscos identificados:**
  - REINFORCE puro com reward esparso (so +/-1 em gols) pode levar dezenas de
    milhares de episodios sem progresso aparente. Mitigacao: reward shaping
    opcional (recompensa por rebatida) no Slice A, ligado no treino real.
  - Diferenca numerica entre PyTorch e onnxruntime-web pode quebrar o
    comportamento aprendido. Mitigacao: teste de paridade dedicado com
    tolerancia `1e-4` no Slice B.
  - FrameStack TS pode divergir do Python na ordem de flatten ou na largura do
    bitmap. Mitigacao: paridade golden sobre o vetor `24000` no Slice C.
  - Tamanho do modelo ONNX em FP32 (~19MB) pode pesar no carregamento.
    Mitigacao: medir; se >25MB, avaliar quantizacao no Slice B.
  - Inferencia no browser pode passar do orcamento de 16ms por frame.
    Mitigacao: rodar inferencia a cada N ticks (ex.: 2 ou 3) em vez de todo
    frame, mantendo a acao anterior entre execucoes.
- **Suposicoes:**
  - 5000-10000 episodios em CPU sao suficientes para o modelo bater random.
  - `onnxruntime-web` consegue rodar este MLP a 60 fps em hardware comum.
  - O motor TS continua bit-a-bit identico ao Python (paridade ja verde).
- **Alternativas rejeitadas:**
  - TensorFlow.js: peso maior, treino mais lento, sem ganho aqui.
  - DQN/PPO: complexidade desproporcional para o objetivo educacional.
  - Self-play: foge do escopo "esquerda heuristica vs direita modelo".

Nenhum problema serio identificado; seguir.

---

## 3. Requisitos

### 3.1 Funcionais

- [ ] Trainer multi-episodio que executa N iteracoes REINFORCE e agrega
  metricas por episodio.
- [ ] Salvar checkpoint `.pt` periodicamente (latest + best por reward medio).
- [ ] Carregar checkpoint `.pt` para retomar treino do ultimo episodio.
- [ ] CLI `training/scripts/train.py` com flags `--episodes`, `--gamma`,
  `--lr`, `--seed`, `--checkpoint-dir`, `--checkpoint-every`,
  `--reward-shaping`.
- [ ] Metricas offline em matplotlib: curva de loss, reward total e
  comprimento de episodio.
- [ ] Reward shaping opcional: bonus por rebatida da raquete direita,
  configuravel e desligado por default.
- [ ] Exportacao ONNX em `training/src/export_onnx.py` que le checkpoint e
  grava `web/public/model.onnx`.
- [ ] Teste de paridade PyTorch <-> ONNX Runtime Python: mesma entrada produz
  logits identicos dentro de tolerancia `1e-4` em pelo menos 50 amostras
  aleatorias.
- [x] FrameStack TypeScript em `web/src/ai/frame_stack.ts` espelhando o
  Python: 5 frames `60x80` em `Uint8Array` -> `Float32Array` de comprimento
  `24000`.
- [x] Teste de paridade FrameStack Python <-> TS sobre fixture golden.
- [x] Inferencia ONNX no browser em `web/src/ai/inference.ts` via
  `onnxruntime-web`: carrega `model.onnx`, expoe
  `forward(obs: Float32Array): Promise<{logits, probs, action}>`.
- [ ] `web/src/main.ts` integra: a cada N ticks, gera bitmap, alimenta o
  FrameStack, chama a inferencia e aplica a acao na raquete direita.
- [ ] Toggle de modo na UI: `model | heuristic | keyboard`. Default = `model`
  se `web/public/model.onnx` esta presente; senao `keyboard` com aviso.
- [ ] Indicador visivel no scoreboard de qual jogador e qual (`heuristica` na
  esquerda, `modelo` na direita).

### 3.2 Nao-funcionais

- **Performance:** inferencia + render < 16ms por frame (60fps) em laptop CPU
  comum.
- **Determinismo:** treino com mesma seed e mesmas flags produz exatamente o
  mesmo checkpoint.
- **Reprodutibilidade:** hiperparametros e seed sao gravados junto com cada
  checkpoint.
- **Tamanho:** `model.onnx` < 25MB em FP32.
- **Compatibilidade:** `onnxruntime-web` e a unica nova dep no browser.
- **Estabilidade:** paridade do motor (`scripts/parity_check.py` e
  `web/tests/parity.test.ts`) continua verde apos cada slice.
- **Robustez:** ausencia de `model.onnx` no browser nao trava o jogo;
  fallback automatico para teclado com aviso.

### 3.3 Fora do escopo

- Visualizacao de ativacoes e pesos da rede (Fase 4 separada).
- DQN, A2C, PPO ou qualquer algoritmo alem de REINFORCE.
- GPU obrigatoria; treino distribuido.
- Self-play (esquerda usar o mesmo modelo).
- Modo mobile com touch.
- Persistencia de partidas, ranking, multi-jogador em rede.

---

## 4. Criterios de Aceitacao

**Caminho feliz:**

- [ ] Dado `--episodes 10` e seed fixa, quando o trainer roda, entao gera 10
  metricas por episodio e salva ao menos 1 checkpoint.
- [ ] Dado checkpoint salvo, quando o exportador ONNX roda, entao gera
  `model.onnx` com tamanho `<25MB`.
- [ ] Dada uma entrada `(1, 24000)` qualquer, quando comparados PyTorch e
  `onnxruntime` Python, entao os logits diferem em menos que `1e-4`.
- [x] Dado FrameStack TS e Python alimentados com a mesma sequencia de 5
  bitmaps, quando o vetor `24000` e comparado, entao e exatamente igual.
- [ ] Dado `model.onnx` presente, quando a pagina carrega, entao a raquete
  direita e controlada pelo modelo (nao pelo teclado) e o jogo roda em 60fps.
- [ ] Dado o modo padrao, quando uma partida e jogada ate 21 pontos, entao
  termina sem erro, sem freeze e sem divergencia visual.

**Casos de borda:**

- [ ] `model.onnx` ausente: jogo carrega em modo `keyboard` com aviso visivel
  na UI; nenhum erro no console.
- [ ] Checkpoint corrompido ou versao incompativel: trainer falha com mensagem
  clara, sem stack-trace nu.
- [x] FrameStack recebe bitmap de shape errado: erro claro com shape esperado
  e shape recebido.
- [ ] Treino interrompido por `Ctrl-C`: salva checkpoint parcial com a
  contagem do ultimo episodio completo.
- [ ] Inferencia mais lenta que 16ms: sistema mantem a acao anterior ate o
  proximo tick de inferencia, sem dropar frame de render.

**Falhas esperadas:**

- [ ] Hiperparametro invalido (`lr<=0`, `gamma<0`, `gamma>1`, `episodes<=0`):
  erro claro antes de iniciar o treino.
- [ ] `onnxruntime-web` falha ao baixar/carregar o modelo: aviso na UI e
  fallback automatico para `heuristic` (espelha a logica da esquerda) ou
  `keyboard`.

---

## 5. Design Tecnico

### 5.1 Abordagem em uma frase

Encerrar o projeto em **4 slices testaveis em sequencia**: trainer multi-ep
+ checkpoints (Python), exportacao ONNX + paridade (Python), FrameStack TS
+ inferencia no browser, e por fim integracao + smoke E2E com modelo real
treinado.

### 5.2 Slices e arquivos

**Slice A - Trainer multi-episodio + Checkpoints + CLI**

| Arquivo | Acao | Motivo |
|---|---|---|
| `training/src/metrics.py` | criar | dataclass `TrainingMetrics` + agregacao |
| `training/src/checkpoint.py` | criar | `save_checkpoint` / `load_checkpoint` |
| `training/src/reward_shaping.py` | criar | bonus por rebatida (opt-in) |
| `training/src/trainer.py` | criar | loop multi-ep reusando `run_reinforce_update` |
| `training/scripts/__init__.py` | criar | tornar `scripts/` importavel |
| `training/scripts/train.py` | criar | CLI argparse |
| `training/tests/test_metrics.py` | criar | TDD |
| `training/tests/test_checkpoint.py` | criar | TDD |
| `training/tests/test_reward_shaping.py` | criar | TDD |
| `training/tests/test_trainer.py` | criar | TDD |
| `PLAN.md` | atualizar | progresso |

**Slice B - Exportacao ONNX + Paridade**

| Arquivo | Acao | Motivo |
|---|---|---|
| `training/src/export_onnx.py` | criar | `torch.onnx.export` com input `(1, 24000)` |
| `training/scripts/export.py` | criar | CLI |
| `training/tests/test_export_onnx.py` | criar | gera, valida, paridade |
| `PLAN.md` | atualizar | progresso |

**Slice C - FrameStack TS + Inferencia ONNX no browser**

| Arquivo | Acao | Motivo |
|---|---|---|
| `web/package.json` | atualizar | add `onnxruntime-web` |
| `shared/fixtures/frame_stack_golden.json` | criar | 5 bitmaps + vetor 24000 esperado |
| `scripts/gen_frame_stack_fixture.py` | criar | gera o golden |
| `web/src/ai/frame_stack.ts` | criar | espelho do Python |
| `web/src/ai/inference.ts` | criar | wrapper `onnxruntime-web` |
| `web/tests/frame_stack.test.ts` | criar | TDD |
| `web/tests/parity_frame_stack.test.ts` | criar | paridade Python<->TS |
| `web/tests/inference.test.ts` | criar | TDD com mock onnxruntime |
| `PLAN.md` | atualizar | progresso |

**Slice D - Integracao browser + smoke E2E + treino real**

| Arquivo | Acao | Motivo |
|---|---|---|
| `web/src/ui/mode_toggle.ts` | criar | toggle `model / heuristic / keyboard` |
| `web/src/ui/scoreboard.ts` | atualizar | rotulos `heuristica` / `modelo` |
| `web/src/main.ts` | atualizar | conecta `inference` no game loop |
| `web/tests/main.test.ts` | criar | smoke em jsdom com inference mock |
| `web/public/model.onnx` | gerado | artefato do treino real |
| `PLAN.md` | atualizar | progresso |
| `docs/plans/2026-04-17-encerramento-treino-e-inferencia-browser.md` | criar | arquivar plan |

### 5.3 Contratos / tipos novos

```python
# training/src/metrics.py
@dataclass(frozen=True)
class TrainingMetrics:
    episode: int
    loss: float
    total_reward: float
    episode_length: int
    avg_loss_window: float        # media movel ultimos 50 episodios
    avg_reward_window: float

# training/src/checkpoint.py
@dataclass(frozen=True)
class Checkpoint:
    state_dict: dict
    optimizer_state: dict
    hparams: dict
    episode: int
    metrics_history: list[dict]

def save_checkpoint(path: Path, ckpt: Checkpoint) -> None: ...
def load_checkpoint(path: Path) -> Checkpoint: ...

# training/src/trainer.py
@dataclass(frozen=True)
class TrainerConfig:
    episodes: int
    gamma: float
    learning_rate: float
    seed: int
    max_steps: int
    checkpoint_dir: Path
    checkpoint_every: int
    reward_shaping: bool = False

def run_training(config: TrainerConfig) -> list[TrainingMetrics]: ...

# training/src/export_onnx.py
def export_to_onnx(model: PongPolicyNetwork, path: Path) -> None: ...
def verify_onnx_parity(
    model: PongPolicyNetwork,
    onnx_path: Path,
    samples: np.ndarray,        # shape (N, 24000)
    tol: float = 1e-4,
) -> bool: ...
```

```ts
// web/src/ai/frame_stack.ts
export interface FrameStackTS {
  push(frame: Uint8Array): void;     // shape H*W = 60*80
  flatten(): Float32Array;            // length 24000
  isReady(): boolean;
  reset(): void;
}

// web/src/ai/inference.ts
export interface InferenceResult {
  readonly logits: Float32Array;
  readonly probs: Float32Array;
  readonly action: Action;
}

export interface ModelInference {
  load(url: string): Promise<void>;
  forward(obs: Float32Array): Promise<InferenceResult>;
  isLoaded(): boolean;
}

// web/src/ui/mode_toggle.ts
export type PlayMode = 'model' | 'heuristic' | 'keyboard';
```

### 5.4 Fluxo de dados

**Treino (Python, headless):**

```
train.py (CLI)
  -> TrainerConfig
  -> run_training:
       loop episodes:
         run_reinforce_update -> ReinforceUpdateResult
         opcional: reward_shaping aplicado dentro do rollout
         agrega TrainingMetrics
         a cada checkpoint_every: save_checkpoint(latest.pt, best.pt)
  -> retorna list[TrainingMetrics]
export.py (CLI)
  -> load_checkpoint
  -> export_to_onnx -> web/public/model.onnx
  -> verify_onnx_parity (em CI / smoke)
```

**Inferencia (browser, 60fps):**

```
gameLoop a cada frame:
  leftAction  = partiallyTracking(state, rng)        # heuristica estocastica
  rightAction = lastModelAction (default: 'none')
  state'      = step(state, leftAction, rightAction)
  draw(state')
  bitmap      = bitmapFromState(state')
  frameStack.push(bitmap)
  if frameStack.isReady() && tick % N === 0:
    obs    = frameStack.flatten()
    result = await inference.forward(obs)
    lastModelAction = result.action
```

### 5.5 Dependencias novas

- **Python:** nenhuma (PyTorch, ONNX, ONNX Runtime, numpy ja declarados).
- **Web:** `onnxruntime-web ^1.17` (justificativa: caminho oficial para
  inferencia ONNX em browser; suporta WASM e WebGL; sem alternativa
  equivalente).

---

## 6. Estrategia de Testes

Ver `TESTING.md`. Para esta feature:

- **Unitarios Python:** trainer (1 e N episodios), checkpoint (roundtrip,
  hparams preservados, erro em path invalido), reward_shaping (com/sem
  rebatida), metrics (janela movel correta), export_onnx (arquivo gerado e
  carregavel).
- **Unitarios TS:** frame_stack (push, flatten length, isReady, reset),
  inference (mock ort.InferenceSession para nao depender de WASM em CI).
- **Paridade:**
  - PyTorch <-> ONNX Runtime Python sobre 50 amostras random (Slice B).
  - FrameStack Python <-> TS sobre golden fixture (Slice C).
  - Motor Python <-> TS continua verde (regressao).
- **Integracao TS:** `main.test.ts` em jsdom com inference mock, 100 ticks
  sem erro (Slice D).
- **Smoke manual:** `pnpm dev` -> partida 0->21 com modelo na direita,
  60fps, sem freeze (Slice D).

---

## 7. Passos de Implementacao

> TDD em todos os passos: teste vermelho -> implementacao minima -> verde.
> Loop de correcao: maximo 3 tentativas por passo antes de escalar.

**Slice A - Trainer + Checkpoints + CLI**

- [x] **A1.** Testes vermelhos para `metrics.py` (agregacao e janela movel).
- [x] **A2.** Implementar `metrics.py`.
- [x] **A3.** Testes vermelhos para `checkpoint.py` (save/load roundtrip,
  hparams preservados, erro em path invalido).
- [x] **A4.** Implementar `checkpoint.py`.
- [x] **A5.** Testes vermelhos para `reward_shaping.py` (com e sem rebatida).
- [x] **A6.** Implementar `reward_shaping.py`.
- [x] **A7.** Testes vermelhos para `trainer.py` (1 episodio, N episodios,
  checkpoint disparado a cada `checkpoint_every`, retomada por
  `load_checkpoint`).
- [x] **A8.** Implementar `trainer.py` reusando `run_reinforce_update`.
- [x] **A9.** Implementar `scripts/train.py` (CLI argparse, validacao de
  flags, mensagens claras de erro).
- [x] **A10.** `uv run ruff check . && uv run pytest`.

**Slice B - Exportacao ONNX + Paridade**

- [x] **B1.** Testes vermelhos para `export_onnx.py` (arquivo gerado, ONNX
  valido, opset esperado).
- [x] **B2.** Implementar `export_onnx.py` com `torch.onnx.export`,
  input dummy `(1, 24000)`.
- [x] **B3.** Implementar `verify_onnx_parity` e teste de paridade
  PyTorch <-> ONNX Runtime sobre 50 amostras random (tolerancia `1e-4`).
- [x] **B4.** Implementar `scripts/export.py` (CLI: `--checkpoint`,
  `--output`).
- [x] **B5.** Treinar curto (50 episodios), exportar, validar paridade
  manualmente.
- [x] **B6.** `uv run ruff check . && uv run pytest`.

**Slice C - FrameStack TS + Inferencia ONNX no browser**

- [x] **C1.** Adicionar `onnxruntime-web` em `web/package.json` e instalar
  com `pnpm install`.
- [x] **C2.** Criar `scripts/gen_frame_stack_fixture.py` e gerar
  `shared/fixtures/frame_stack_golden.json` (5 bitmaps + vetor 24000
  esperado).
- [x] **C3.** Testes vermelhos para `web/src/ai/frame_stack.ts` (push,
  flatten length, isReady, reset, erro em shape invalido).
- [x] **C4.** Implementar `frame_stack.ts`.
- [x] **C5.** Implementar e rodar `parity_frame_stack.test.ts` sobre o
  golden.
- [x] **C6.** Testes vermelhos para `inference.ts` com `ort.InferenceSession`
  mockado.
- [x] **C7.** Implementar `inference.ts` com `onnxruntime-web` (carregamento
  lazy, cache, erro tratado).
- [x] **C8.** `pnpm --filter web typecheck && pnpm --filter web lint &&
  pnpm --filter web test`.

**Slice D - Integracao browser + smoke E2E + treino real**

- [x] **D1.** Criar `web/src/ui/mode_toggle.ts` (`model / heuristic /
  keyboard`); default `model` se ONNX presente, senao `keyboard` com aviso.
- [x] **D2.** Atualizar `scoreboard.ts` para rotular `heuristica` (esquerda)
  e `modelo` (direita).
- [x] **D3.** Atualizar `main.ts`: integrar `frameStack` + `inference` no
  game loop; rodar inferencia a cada N ticks; manter ultima acao entre
  ticks; tratar `model.onnx` ausente com fallback.
- [x] **D4.** Criar `web/tests/main.test.ts` (smoke em jsdom com `inference`
  mockado, 100 ticks sem erro).
- [x] **D5.** Treinar modelo real (`pnpm` n/a; rodar `uv run python
  training/scripts/train.py --episodes 10000 --reward-shaping --seed 42
  --device cuda --batch-envs 8`).
- [x] **D6.** Exportar e copiar `model.onnx` para `web/public/`.
- [ ] **D7.** `pnpm --filter web dev` -> smoke manual: jogo roda 60fps,
  modelo joga, partida 0->21 termina sem erro.
- [ ] **D8.** Verificar todos os criterios da secao 4 (caminho feliz,
  bordas, falhas).
- [x] **D9.** Suite completa: `uv run ruff check . && uv run pytest && pnpm
  --filter web typecheck && pnpm --filter web lint && pnpm --filter web
  test && python scripts/parity_check.py`.
- [x] **D10.** Arquivar `PLAN.md` em
  `docs/plans/2026-04-17-encerramento-treino-e-inferencia-browser.md`.

**Execucao atual solicitada pelo usuario (fechamento com treino curto otimizado)**

- [x] **D5.100.a** Rodar treino otimizado de 100 episodios com CUDA +
  batching (`--device cuda --batch-envs 8`) e salvar checkpoints.
- [x] **D6.100.b** Exportar o checkpoint treinado para
  `web/public/model.onnx`.
- [x] **D7.100.c** Subir o dev server com o modelo exportado disponivel e
  registrar URL/porta.
- [x] **D8.100.d** Executar verificacoes finais automatizadas e registrar o
  que ficou pendente de smoke visual/manual.
- [x] **D10.100.e** Arquivar o `PLAN.md` atual em `docs/plans/`.

**Resultado desta execucao curta (2026-04-17)**

- [x] Treino otimizado executado com
  `uv run python training/scripts/train.py --episodes 100 --reward-shaping --seed 42 --device cuda --batch-envs 8 --checkpoint-dir training/checkpoints/final-100`.
- [x] Resultado final do treino: `final_loss=-0.0320`,
  `final_reward=-0.80`, `avg_reward_window=-1.66`.
- [x] Checkpoints gerados em `training/checkpoints/final-100/latest.pt` e
  `training/checkpoints/final-100/best.pt`.
- [x] Exportacao ONNX executada com
  `uv run python training/scripts/export.py --checkpoint training/checkpoints/final-100/best.pt --output web/public/model.onnx`.
- [x] Paridade PyTorch <-> ONNX validada em 50 amostras:
  `max_abs_diff=1.490e-07`, dentro da tolerancia `1e-4`.
- [x] `web/public/model.onnx` gerado com `19,444,909` bytes
  (aprox. `18.54 MiB`), abaixo do limite de `25MB`.
- [x] Dev server web respondeu em `http://localhost:5173/` e o artefato
  `http://localhost:5173/model.onnx` retornou `HTTP 200`.
- [x] Checks automatizados verdes:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.
- [ ] Pendente: smoke visual/manual do browser para confirmar partida real
  `0 -> 21`, comportamento perceptivel do modelo e ausencia de erros no
  console durante uma sessao completa.

**Plano detalhado desta execucao (slice D parcial ate smoke browser):**

- [x] **D1.a** Modelar contrato de modo (`model | heuristic | keyboard`) e
  estado de fallback/aviso em modulo de UI proprio.
- [x] **D1.b** Atualizar `scoreboard.ts` para aceitar metadados de modo e
  expor texto de status deterministico.
- [x] **D1.c** Refatorar `main.ts` para uma funcao inicializadora testavel,
  isolando DOM, `requestAnimationFrame`, `FrameStack` e `ModelInference`.
- [x] **D1.d** Conectar bitmap + frame stack + inferencia com throttling por
  `INFERENCE_INTERVAL_TICKS`, reaproveitando a ultima acao entre inferencias.
- [x] **D1.e** Implementar fallback automatico quando `model.onnx` nao
  existir ou falhar ao carregar, sem travar o jogo.
- [x] **D1.f** Criar smoke test do app com DOM simulado e inferencia mockada,
  cobrindo 100 ticks sem erro.
- [x] **D1.g** Rodar checks web completos e, se possivel, iniciar o dev
  server para smoke manual.

**Status do treino interrompido manualmente:**

- [ ] **D5.a** Treino real de 5000 episodios iniciado em `2026-04-17`, mas
  interrompido manualmente antes de concluir.
- [x] **D5.b** Checkpoints parciais gerados durante a execucao:
  `training/checkpoints/latest.pt` e `training/checkpoints/best.pt`.
- [ ] **D5.c** Retomar o treino a partir de `training/checkpoints/latest.pt`
  ou reiniciar a execucao completa antes da exportacao final ONNX.

**Slice D - preparacao para treino em GPU NVIDIA**

- [x] **D5.gpu1** Verificar driver NVIDIA, visibilidade da GPU e se o `torch`
  do ambiente atual tem suporte CUDA.
- [x] **D5.gpu2** Adaptar o trainer para selecionar dispositivo
  (`auto | cpu | cuda`) e registrar o device usado nos hparams.
- [x] **D5.gpu3** Ajustar CLI de treino para expor `--device`.
- [x] **D5.gpu4** Garantir carga de checkpoint com `map_location`
  compativel entre CPU e CUDA.
- [x] **D5.gpu5** Instalar build CUDA do PyTorch no ambiente `uv` do projeto.
- [x] **D5.gpu6** Validar com teste real: `torch.cuda.is_available()`,
  nome da GPU e operacao simples de tensor na GPU.

**Resultado da validacao GPU (2026-04-17):**

- [x] Driver NVIDIA detectado via `nvidia-smi`: `581.95`; GPU:
  `NVIDIA GeForce RTX 3070 Laptop GPU`.
- [x] Ambiente Python do projeto atualizado para `torch 2.11.0+cu128`.
- [x] `torch.cuda.is_available()` retornou `True` e `torch.version.cuda`
  retornou `12.8`.
- [x] Operacao real em GPU validada com multiplicacao de matrizes em
  `cuda:0`.
- [x] Smoke real do trainer validado com
  `uv run python training/scripts/train.py --episodes 1 --reward-shaping --seed 42 --device cuda --checkpoint-dir training/checkpoints/gpu-smoke`.

**Slice D - otimizacao do hot loop de treino**

- [x] **D5.opt1** Red tests para garantir que `FrameStack` continua com o
  mesmo contrato logico apos trocar deque por buffer circular pre-alocado.
- [x] **D5.opt2** Reimplementar `training/src/frame_stack.py` com buffer
  fixo `(5, 60, 80)`, indice circular e flat buffer reutilizavel.
- [x] **D5.opt3** Eliminar alocacao recorrente do bitmap no treino via buffer
  reutilizavel por episodio/update, sem tocar no motor canonico do `engine/`.
- [x] **D5.opt4** Ajustar `rollout.py` e `reinforce.py` para usar os buffers
  reutilizaveis mantendo determinismo e contrato dos testes.
- [x] **D5.opt5** Rodar `uv run ruff check . && uv run pytest`.
- [x] **D5.opt6** Medir novamente 10 episodios com
  `--reward-shaping --seed 42 --device cuda` e comparar com o baseline de
  `41.047s` coletado antes da otimizacao.

**Resultado da medicao pos-otimizacao (2026-04-17):**

- [x] Baseline anterior de 10 episodios: `41.047s`.
- [x] Nova medicao de 10 episodios apos buffers reutilizaveis: `35.135s`.
- [x] Ganho absoluto: `5.912s` a menos em 10 episodios.
- [x] Ganho relativo: `14.40%` mais rapido (`1.168x`).
- [x] Nova estimativa para 5000 episodios: `17567.6s` (`4.88h`).

**Slice D - fast path de treino**

- [x] **D5.fast1** Red tests para garantir que o caminho rapido preserva o
  contrato logico do `FrameStack` e o treino segue deterministico.
- [x] **D5.fast2** Permitir `FrameStack` sem buffer de debug no hot loop do
  treino.
- [x] **D5.fast3** Adicionar caminho rapido para observacao `float32`
  reutilizavel, evitando `astype(..., copy=False)` a cada tick do treino.
- [x] **D5.fast4** Ajustar `reinforce.py` para usar o caminho rapido sem
  validacao/copia redundante no loop.
- [x] **D5.fast5** Rodar `uv run ruff check . && uv run pytest`.
- [x] **D5.fast6** Medir novamente 10 episodios com
  `--reward-shaping --seed 42 --device cuda` e comparar com o baseline atual
  de `35.135s`.

**Resultado da medicao pos-fast-path (2026-04-17):**

- [x] Baseline anterior de 10 episodios: `35.135s`.
- [x] Nova medicao de 10 episodios apos fast path: `32.900s`.
- [x] Ganho absoluto: `2.235s` a menos em 10 episodios.
- [x] Ganho relativo: `6.36%` mais rapido (`1.068x`).
- [x] Nova estimativa para 5000 episodios: `16450s` (`4.57h`).

**Slice D - estado interno otimizado para treino**

- [x] **D5.state1** Red tests para equivalencia entre estado interno do treino
  e o engine canonico em criacao inicial, heuristica e sequencia curta de
  passos.
- [x] **D5.state2** Criar estado mutavel leve para treino com API interna
  propria, sem `frozen dataclass` e sem `__post_init__` no hot loop.
- [x] **D5.state3** Ajustar `reinforce.py` para usar o estado interno no loop
  quente, mantendo o comportamento atual do treino.
- [x] **D5.state4** Rodar `uv run ruff check . && uv run pytest`.
- [x] **D5.state5** Medir novamente 10 episodios com
  `--reward-shaping --seed 42 --device cuda` e comparar com o baseline atual
  de `32.900s`.

**Resultado da tentativa com estado interno mutavel (2026-04-17):**

- [x] Baseline anterior de 10 episodios: `32.900s`.
- [x] Medicao da tentativa com estado interno mutavel: `33.755s`.
- [x] Resultado: regressao de `0.855s` em 10 episodios (`2.60%` mais lento).
- [x] Decisao de engenharia: reverter a tentativa e manter o caminho
  anterior, que segue mais rapido e mais simples.

**Slice D - batch de multiplos ambientes**

- [x] **D5.batch1** Implementar caminho de treino com multiplos ambientes em
  paralelo, mantendo `episodes` como total de episodios processados.
- [x] **D5.batch2** Agrupar inferencia do modelo em batch e manter os estados
  de treino em arrays NumPy.
- [x] **D5.batch3** Expor `--batch-envs` na CLI de treino.
- [x] **D5.batch4** Cobrir com testes dedicados e rodar
  `uv run ruff check . && uv run pytest`.
- [x] **D5.batch5** Medir novamente 10 episodios e comparar com o baseline de
  `32.900s`, validando tambem `batch_envs=4` e `batch_envs=8`.

**Resultado da medicao pos-batching (2026-04-17):**

- [x] Baseline anterior de 10 episodios: `32.900s`.
- [x] Medicao com `batch_envs=4`: `23.952s` (`27.20%` mais rapido).
- [x] Medicao com `batch_envs=8`: `20.040s` (`39.09%` mais rapido).
- [x] Melhor configuracao medida ate aqui: `batch_envs=8`.
- [x] Nova estimativa para 5000 episodios com `batch_envs=8`:
  `10020s` (`2.78h`).

---

## 8. Rollout e Reversibilidade

- **Feature flag:** modo de jogo (`model | heuristic | keyboard`) controlado
  por toggle UI. Default `model` se `web/public/model.onnx` existe; senao
  `keyboard`.
- **Migracao de dados:** nenhuma.
- **Plano de rollback:**
  - Remover `web/public/model.onnx` reverte automaticamente para fallback.
  - Descartar checkpoints `.pt` reverte o treino sem afetar o codigo.
  - Reverter o branch desfaz toda a integracao.
- **Impacto em usuarios existentes:** nenhum (projeto local, sem deploy).

---

## 9. Checklist Final

- [ ] Todos os criterios da secao 4 tem teste verde (ou validacao manual
  registrada).
- [ ] `uv run ruff check . && uv run pytest` passa.
- [ ] `pnpm --filter web typecheck && pnpm --filter web lint && pnpm
  --filter web test` passa.
- [ ] `python scripts/parity_check.py` passa.
- [ ] Smoke manual: partida 0->21 com modelo na direita roda em 60fps sem
  freeze nem erro no console.
- [ ] `web/public/model.onnx` existe e tem `<25MB`.
- [ ] Tamanho do bundle web nao regrediu mais que 5% alem do `model.onnx`.
- [ ] Nenhum `TODO`/`FIXME` novo sem issue vinculada.
- [ ] Commits seguem Conventional Commits (`feat`, `test`, `chore` por
  slice).
- [ ] `PLAN.md` arquivado em `docs/plans/`.

---

## 10. Log de Decisoes (append-only)

- **2026-04-17** - encerrar o projeto em 4 slices A->B->C->D em vez de um
  plano monolitico. Motivo: cada slice e parado entre si, reduz risco e
  permite parar entre slices se algo divergir.
- **2026-04-17** - reward shaping (bonus por rebatida) entra como feature
  opt-in no Slice A e e ligado no treino real do Slice D. Motivo: REINFORCE
  puro com reward esparso (so +/-1 em gols) demanda muitos episodios;
  rebatida acelera convergencia sem mudar o algoritmo.
- **2026-04-17** - exportacao para ONNX em vez de TF.js. Motivo: PyTorch e
  fonte canonica do treino; `onnxruntime-web` e o caminho oficial e suporta
  WASM/WebGL.
- **2026-04-17** - inferencia rodada a cada N ticks (nao todo frame) com
  cache da ultima acao. Motivo: protege o orcamento de 16ms/frame mesmo se
  o forward levar mais tempo em maquinas mais lentas.
- **2026-04-17** - visualizacao de ativacoes/pesos fica fora deste plano e
  vai para uma "Fase 4" futura. Motivo: o objetivo declarado pelo usuario
  agora e "jogo rodando com heuristica vs modelo treinado"; visualizacao
  amplia escopo sem ser bloqueante para esse fim.
