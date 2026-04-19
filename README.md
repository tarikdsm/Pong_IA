# Pong IA

Pong 2D para browser com motor deterministico espelhado em Python e TypeScript,
treino em PyTorch e inferencia ONNX no frontend.

Hoje o projeto ja entrega:

- jogo completo no browser com seletores independentes para as duas raquetes
- modos `modelo`, `heuristica` e `teclado`
- treino headless em Python com checkpoints e exportacao ONNX
- paridade automatizada entre o motor Python e o port TypeScript
- painel dedicado no browser para visualizar frames de entrada, ativacoes e
  pesos da primeira camada do modelo

O modelo publicado no jogo atualmente vem de
`training/checkpoints/tune-from-best-2000-centering/best.pt` e e servido em
`web/public/model.onnx`.

## Estado Atual

- **Raquete esquerda:** `modelo | heuristica | teclado`
- **Raquete direita:** `modelo | heuristica | teclado`
- **Teclado:** esquerda em `W/S`, direita em `ArrowUp/ArrowDown`
- **Cadencia do modelo:** 30 ticks (~0,5 s)
- **Cadencia da heuristica:** 60 ticks (~1,0 s)
- **Observacao do modelo:** 5 frames binarios com stride temporal fixo

## Estrutura

```text
pong_ia/
|- AGENTS.md
|- ARCHITECTURE.md
|- PLAN.md
|- README.md
|- TESTING.md
|- WORKFLOW.md
|- docs/
|  |- adr/
|  |- plans/
|  `- references/
|- engine/
|  |- pong_engine/
|  `- tests/
|- training/
|  |- scripts/
|  |- src/
|  |- tests/
|  |- checkpoints/
|  `- reports/
|- web/
|  |- public/
|  |- src/
|  `- tests/
|- shared/
|  |- config.json
|  `- fixtures/
`- scripts/
```

## Requisitos

- Python 3.11+ com `uv`
- Node.js 20+ com `pnpm`

## Setup

```bash
uv sync
pnpm install
python scripts/gen_config.py
python scripts/gen_fixture.py
python scripts/parity_check.py
```

## Rodar o Jogo

```bash
pnpm --filter web dev
```

Abra `http://localhost:5173`.

## Publicar no GitHub Pages

O frontend ficou preparado para GitHub Pages, inclusive quando o site e servido
em subcaminho de repositorio como
`https://tarikdsm.github.io/Pong_IA/`.

O workflow de deploy esta em
`.github/workflows/deploy-pages.yml`.

Passos no GitHub:

1. subir o repositorio
2. abrir `Settings > Pages`
3. em `Build and deployment`, selecionar `GitHub Actions`
4. fazer push na branch `main`

O deploy publica o conteudo de `web/dist`, incluindo `model.onnx`,
`model-viz.json` e `model-first-layer.uint8.bin`.

Os arquivos ativos em `web/public/` ficam versionados no repositorio para que o
deploy no GitHub Pages consiga publicar o frontend ja com o modelo pronto.

## Treinar o Modelo

Treino curto:

```bash
uv run python training/scripts/train.py --episodes 100 --device cuda --batch-envs 8
```

Treino mais robusto:

```bash
uv run python training/scripts/train.py --episodes 2000 --device cuda --batch-envs 8 --evaluation-interval 100 --evaluation-episodes 100
```

Exportar ONNX:

```bash
uv run python training/scripts/export.py --checkpoint training/checkpoints/tune-from-best-2000-centering/best.pt --output web/public/model.onnx
```

Esse comando tambem gera `web/public/model-viz.json` e
`web/public/model-first-layer.uint8.bin`, usados pela visualizacao no browser.

Benchmark curto:

```bash
uv run python training/scripts/benchmark_training.py --episodes 10 --target-episodes 5000 --device cuda --batch-envs 8
```

## Debug de Bitmaps

Gerar exemplos de observacao:

```bash
uv run python training/scripts/dump_debug_bitmaps.py
```

Converter `.pgm` para `.png`:

```bash
uv run python training/scripts/convert_debug_bitmaps.py
```

## Validacao

```bash
uv run ruff check .
uv run pytest
python scripts/parity_check.py
pnpm --filter web typecheck
pnpm --filter web lint
pnpm --filter web test
```

## Documentacao

- [AGENTS.md](AGENTS.md): regras operacionais do projeto
- [ARCHITECTURE.md](ARCHITECTURE.md): arquitetura e ADR-lite
- [TESTING.md](TESTING.md): contrato de testes
- [WORKFLOW.md](WORKFLOW.md): ciclo de trabalho
- [PLAN.md](PLAN.md): plano ativo e pendencias de fechamento
- [docs/plans](docs/plans): historico de execucoes anteriores
- [docs/references/Architecting_and_Visualizing_Deep_Reinforcement_Le.pdf](docs/references/Architecting_and_Visualizing_Deep_Reinforcement_Le.pdf): paper de referencia

## Pendencias Abertas

- consolidar um checklist final de release/entrega
- decidir se ainda entra mais uma rodada de ajuste fino de treino antes do
  fechamento definitivo

## Licenca

Este projeto esta sob a licenca [MIT](LICENSE).
