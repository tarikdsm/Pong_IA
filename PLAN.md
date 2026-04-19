# PLAN.md

> Arquivo vivo. Uma tarefa por vez. Historico antigo foi arquivado em
> `docs/plans/2026-04-19-historico-operacional-ate-cleanup.md`.

---

## 0. Metadados

- **Feature / Tarefa:** Visualizacao de ativacoes e pesos no browser
- **Slug:** `browser-model-visualization`
- **Data de inicio:** 2026-04-19
- **Status:** Em andamento

---

## 1. Critica Rapida

- A visualizacao precisa entrar sem regredir o que o usuario ja aprovou: nada de HUD poluindo o canvas ou quebrando o layout responsivo.
- Hoje o browser so recebe `logits` do ONNX. Para mostrar ativacoes reais e pesos interpretaveis, precisamos exportar saídas intermediarias e um artefato auxiliar da primeira camada.
- A implementacao precisa continuar robusta quando o artefato de visualizacao nao estiver disponivel: o jogo e a inferencia principal nao podem parar por causa disso.

---

## 2. Passos

- [x] **VIZ.1** Atualizar o pipeline de export para publicar `logits`, ativacoes intermediarias e um artefato auxiliar da primeira camada.
- [x] **VIZ.2** Ajustar a inferencia web para consumir as novas saidas sem quebrar fallback e compatibilidade operacional.
- [x] **VIZ.3** Criar um painel dedicado no browser para exibir frames de entrada, ativacoes e pesos do neuronio mais ativado.
- [x] **VIZ.4** Cobrir export e frontend com testes automatizados de regressao.
- [x] **VIZ.5** Reexportar o modelo atual para a web e validar a stack completa.
- [x] **VIZ.6** Melhorar o zoom/layout responsivo do frontend para acomodar o painel dedicado em janelas menores.
- [x] **DEPLOY.1** Ajustar frontend e build para funcionar em GitHub Pages, inclusive em subcaminho de repositorio.
- [x] **DEPLOY.2** Adicionar workflow/documentacao de deploy no GitHub Pages.
- [x] **DEPLOY.3** Inicializar git local, conectar ao repositorio remoto e publicar no GitHub sem perder a validacao local.

---

## 3. Resultado Esperado

- O browser passa a mostrar a "visao" do modelo sem usar HUD sobre o jogo.
- O artefato ONNX continua servindo a inferencia normal, agora com saidas suficientes para a visualizacao.
- Os pesos da primeira camada ficam disponiveis em formato leve para visualizacao interpretavel.
- A feature entra com testes e sem regressao do layout responsivo.

---

## 4. Resultado

- O export do modelo agora publica `logits`, `hidden_one`, `hidden_two` e
  `hidden_three` no ONNX.
- O pipeline de export passou a gerar tambem `web/public/model-viz.json` e
  `web/public/model-first-layer.uint8.bin`, com pesos quantizados da primeira
  camada para visualizacao no browser.
- A web ganhou um painel dedicado, abaixo do jogo, que mostra:
  - os 5 frames de entrada efetivamente consumidos pelo modelo
  - as ativacoes das 3 camadas ocultas
  - os pesos do neuronio mais ativado da primeira camada
- O frontend agora aplica escala automatica do layout para encaixar jogo e
  painel dedicado no viewport sem depender de rolagem vertical da pagina.
- O frontend e o build ficaram compatíveis com GitHub Pages usando caminhos
  relativos para `model.onnx`, `model-viz.json` e
  `model-first-layer.uint8.bin`, inclusive em subcaminho de repositorio.
- O repositório foi inicializado localmente, publicado em
  `https://github.com/tarikdsm/Pong_IA` e configurado com Pages em modo
  `workflow`.
- A feature degrada com graca quando o artefato auxiliar nao existe: o jogo
  segue funcionando e a visualizacao mostra indisponibilidade sem derrubar a
  inferencia.
- O modelo atual foi reexportado a partir de
  `training/checkpoints/tune-from-best-2000-centering/best.pt`.
- Validacoes verdes apos a entrega:
  `uv run ruff check .`, `uv run pytest`, `python scripts/parity_check.py`,
  `pnpm --filter web typecheck`, `pnpm --filter web lint`,
  `pnpm --filter web test`.

---

## 5. O Que Falta Para Finalizar

- Consolidar um checklist final de release/entrega.
- Decidir se ainda entra mais uma rodada de ajuste fino de treino antes do
  fechamento definitivo.

---

## 6. Atualizacao Manual

- Smoke manual completo ja foi executado e validado no browser com
  `teclado`, `heuristica` e `modelo`.
- A visualizacao agora faz parte oficial do escopo final do projeto.
