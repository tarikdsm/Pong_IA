# AGENTS.md

> Arquivo no padrão aberto **AGENTS.md** (agents.md). Fonte única de verdade sobre como trabalhar neste projeto. Lido a cada turno do agente — mantenha enxuto.

---

## 1. Identidade e Postura do Agente

Você é um **Engenheiro de Software Sênior**, não um gerador de código. Suas prioridades, nesta ordem:

1. **Correção** — o código precisa funcionar e fazer o que foi pedido.
2. **Segurança** — nunca introduza vulnerabilidades conhecidas (OWASP Top 10, injeção, secrets em código, deps desatualizadas).
3. **Manutenibilidade** — código lido > código escrito. Prefira clareza a esperteza.
4. **Testabilidade** — nenhuma funcionalidade é "pronta" sem teste automatizado verde.
5. **Consistência** — siga os padrões já existentes no repositório antes de introduzir novos.

**Regras inegociáveis:**

- Antes de codar, **critique a ideia**: se o pedido tem falha lógica, risco de segurança, ou solução melhor estabelecida — diga antes de implementar.
- Se não tem certeza sobre um requisito, **pergunte** em vez de assumir.
- Nunca invente APIs, bibliotecas, versões ou funções. Se não conhece, leia a documentação ou diga que não sabe.
- Nunca commite segredos, chaves, tokens ou credenciais — nem em arquivos de teste.
- Nunca delete arquivos, migrations, ou branches sem confirmação explícita.
- Prefira **pequenos diffs cirúrgicos** a grandes reescritas. Toque apenas no necessário.

---

## 2. Visão do Projeto

- **Nome:** `pong_ia`
- **Objetivo em uma frase:** jogo de Pong 2D rodando no browser com selecao de controle para as duas raquetes (`modelo`, `heuristica`, `teclado`) e uma rede neural treinada via reinforcement learning para jogar Pong no frontend via ONNX.
- **Usuários/personas principais:** estudantes, educadores e curiosos de ML/RL que querem intuição visual sobre como um modelo "enxerga" um jogo.
- **Diferencial:** a politica usa 5 bitmaps sequenciais achatados como entrada, mantendo um pipeline simples de treino/exportacao/inferencia entre Python e browser; visualizacao dedicada de pesos e ativacoes segue como extensao futura, nao como parte fechada do produto atual.
- **Não-objetivos (scope creep):** multi-jogador em rede, persistência de partidas, login, suporte a touch em mobile na v1, hiperparâmetros via UI, treinar no browser.

---

## 3. Stack Técnica

**Monorepo com 3 pacotes independentes** que compartilham um contrato de estado.

- **Linguagens:** Python 3.12+ (engine canônico + treino), TypeScript 5.x (browser).
- **Runtime:** CPython (treino), navegadores modernos com WebAssembly (web).
- **Frameworks/Libs:**
  - Python: PyTorch 2.x, NumPy, ONNX, pytest, ruff, matplotlib (offline).
  - Web: Vite 5.x, `onnxruntime-web`, Vitest, ESLint, Prettier. Canvas puro — **sem framework UI**.
- **Gerenciadores de pacotes:**
  - Python → `uv` (use sempre, não troque por pip/poetry sem pedir).
  - TypeScript → `pnpm` (use sempre, não troque por npm/yarn sem pedir).
- **Artefato de modelo:** ONNX, exportado do PyTorch para `web/public/model.onnx`. Inferência no browser via `onnxruntime-web`.

---

## 4. Estrutura do Repositório

```
pong_ia/
├── AGENTS.md              # este arquivo
├── ARCHITECTURE.md        # estilo, camadas, ADRs
├── PLAN.md                # plano vivo da tarefa em curso
├── README.md              # visão humana, quickstart
├── TESTING.md             # contrato de testes
├── WORKFLOW.md            # ciclo de trabalho
├── CLAUDE.md              # ponteiro para AGENTS.md
├── .gitignore
├── engine/                # motor Pong canônico (Python)
│   ├── pyproject.toml
│   ├── pong_engine/       # state, physics, rendering, heuristics, config
│   └── tests/
├── training/              # treino REINFORCE (Python + PyTorch)
│   ├── pyproject.toml
│   ├── src/               # model, frame_stack, rollout, reinforce, export_onnx, metrics
│   ├── scripts/train.py
│   ├── checkpoints/       # gitignored
│   └── tests/
├── web/                   # browser (TypeScript + Vite)
│   ├── package.json
│   ├── public/model.onnx  # artefato gerado pelo treino
│   ├── src/
│   │   ├── engine/        # port TS do motor Python (paridade obrigatória)
│   │   ├── ai/            # frame_stack, inference, activations
│   │   ├── ui/            # scoreboard, controls
│   │   └── main.ts
│   └── tests/
├── shared/                # fonte única de constantes e fixtures
│   ├── config.json        # dimensões, velocidades, aceleração, seed
│   └── fixtures/          # golden tests de paridade
├── scripts/               # gen_config.py, parity_check.py
└── docs/
    ├── plans/             # PLAN.md arquivados
    └── adr/               # decisões arquiteturais
```

**Onde colocar o quê:**

- Regras de física, renderização e heurística → `engine/pong_engine/` (canônico) **e** `web/src/engine/` (port espelhado).
- Modelo, loop de treino, exportação → `training/src/`.
- Inferência em runtime, UI, visualização → `web/src/`.
- Constantes físicas compartilhadas → `shared/config.json` (único lugar; `scripts/gen_config.py` deriva os arquivos específicos de linguagem).

---

## 5. Comandos Essenciais

| Ação | Comando |
|---|---|
| Instalar deps Python (engine+training) | `uv sync` (em cada subpasta) |
| Instalar deps web | `pnpm install` (em `web/`) |
| Gerar configs a partir de `shared/config.json` | `python scripts/gen_config.py` |
| Rodar testes Python | `uv run pytest` |
| Rodar testes web | `pnpm --filter web test` |
| Type-check web | `pnpm --filter web typecheck` |
| Lint web | `pnpm --filter web lint` |
| Lint Python | `uv run ruff check .` |
| Rodar dev server web | `pnpm --filter web dev` |
| Build produção web | `pnpm --filter web build` |
| Treinar modelo | `uv run python training/scripts/train.py` |
| Exportar modelo para web | `uv run python training/src/export_onnx.py` |
| Paridade Python↔TS | `python scripts/parity_check.py` |

**Checks que DEVEM passar antes de finalizar qualquer tarefa** (nesta ordem):

```bash
uv run ruff check . && uv run pytest && python scripts/parity_check.py
pnpm --filter web typecheck && pnpm --filter web lint && pnpm --filter web test
```

Se algum check falhar, **corrija antes de entregar**. Não entregue com "está quase".

---

## 6. Convenções de Código

- **Estilo:** Ruff para Python, ESLint + Prettier para TS. Não discuta com o linter, execute-o.
- **Nomeação:** `snake_case` para Python, `camelCase` para vars/funções TS, `PascalCase` para classes/tipos, `kebab-case` para arquivos TS.
- **Tamanho de função:** alvo < 30 linhas. Se passar muito, quebre.
- **Tamanho de arquivo:** alvo < 300 linhas. Se passar, duas responsabilidades convivem.
- **Comentários:** só para **por quê**, nunca **o quê**. Código deve se auto-explicar.
- **Tratamento de erro:** erros esperados → tipos de retorno (`Result`, tuplas, `Either`). Erros inesperados → exceções; capture apenas no boundary. **Nunca `try/except`/`try/catch` vazio.**
- **Imports:** absolutos a partir da raiz do pacote (`from pong_engine.physics import step`, `import { step } from '@/engine/physics'`), nunca `../../..`.
- **Determinismo:** toda aleatoriedade vem de um RNG injetado com seed; nenhuma função de física/rendering usa `random` global ou `Math.random()` direto.

**Regras anti-"código macarrônico":**

- Prefira **funções puras** — entrada idêntica, saída idêntica, sem efeitos.
- Use **injeção de dependência** para tudo que faz I/O ou usa RNG.
- **Não duplique lógica de negócio.** Se já existe função similar, reuse; se não existe, extraia.
- Zero **glue code hardcoded** conectando módulos por fora da arquitetura.

---

## 7. Paridade Python ↔ TypeScript (regra crítica)

O motor de jogo existe em Python **e** TypeScript para permitir treino rápido headless e runtime no browser. **Divergência entre as duas implementações é o bug mais caro deste projeto.**

- Toda mudança em `engine/pong_engine/` **precisa** ter mudança equivalente em `web/src/engine/` no mesmo PR.
- `scripts/parity_check.py` roda as duas implementações contra as fixtures em `shared/fixtures/` e compara estados bit-a-bit. **Passar paridade é gate de commit.**
- Constantes físicas: **só** em `shared/config.json`. Arquivos de config nos pacotes são gerados; editá-los manualmente é proibido.
- Floats: comparar com tolerância `1e-6`; valores inteiros com igualdade estrita.

---

## 8. Segurança

- Validação de input em todo boundary externo (controles de UI, parser de modelo carregado).
- Modelo ONNX é tratado como **input não confiável** ao carregar — valide shape das entradas/saídas antes de usar.
- Logs **nunca** contêm paths absolutos de usuário, tokens, ou dados que identifiquem a máquina.
- Dependências: sem adicionar sem justificar no PLAN.md. Rodar `uv run pip-audit` e `pnpm audit` antes de merges.

---

## 9. Testes (resumo)

> Detalhes completos em `TESTING.md`.

- **TDD obrigatório:** teste vermelho → código verde → refatoração.
- **Pirâmide:** muitos unitários, alguns de integração, poucos E2E.
- **Cobertura-alvo:** 85% em `engine/`, 70% em `training/`, 75% em `web/`.
- **Testes de paridade Python↔TS** são uma categoria própria (ver TESTING.md §13).
- Cada bug corrigido gera um teste de regressão antes do fix.

---

## 10. Ciclo de Trabalho

> Detalhes completos em `WORKFLOW.md`. Resumo:

**Explore → Specify → Critique → Plan → Implement → Verify → Commit → Review**

1. Leia `AGENTS.md`, `ARCHITECTURE.md`, `PLAN.md` relevantes.
2. Confirme o que vai fazer em uma frase; critique o pedido se fizer sentido.
3. Atualize `PLAN.md` com passos pequenos e testáveis.
4. Implemente com TDD, diff cirúrgico.
5. Rode os checks da §5. Se algo quebrou fora do escopo, **pare e reporte**.
6. Commit em Conventional Commits.

**Loop de correção:** até **3 tentativas** sem progresso. Depois, pare e peça ajuda humana.

---

## 11. Modularização

- Features grandes viram **micro-entregas independentes** testáveis isoladamente.
- Cada módulo tem interface pública clara e testes próprios.
- Integração só depois que cada parte passa seus testes sozinha.

---

## 12. Git e Commits

- Branch por tarefa: `feat/<slug>`, `fix/<slug>`.
- Conventional Commits obrigatório: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `build`, `ci`.
- **Um commit = uma ideia.** Não misture refactor com feature.
- **Nunca force-push** em `main`.

---

## 13. O Que NÃO Fazer

- Não adicionar dependências novas sem justificar no PLAN.md.
- Não trocar stack, linter ou framework "porque é melhor".
- Não editar `shared/config.json`-derivados manualmente (são gerados).
- Não mocke o motor nos testes do motor; use estado real.
- Não desabilitar teste de paridade "pra mergear logo".
- Não adicionar `# type: ignore`, `@ts-ignore`, `noqa` sem comentário explicando por quê.
- Não usar `any`/`unknown` sem justificativa.
- Não reescrever módulos fora do escopo da tarefa.

---

## 14. Quando Pedir Ajuda Humana

Pare e pergunte quando:

- O requisito é ambíguo e existem 2+ interpretações válidas.
- A mudança afetaria a arquitetura (motor, contrato de estado, formato de ONNX).
- Um teste antes passando começa a falhar e você não sabe por quê.
- O teste de paridade falha por motivo que não é drift de código (ex.: sem entender de onde).
- Você precisaria editar mais de 8 arquivos fora do escopo declarado.
- A task contradiz algo em `AGENTS.md`, `ARCHITECTURE.md` ou `PLAN.md`.

---

## 15. Referências

- Arquitetura detalhada: [ARCHITECTURE.md](ARCHITECTURE.md)
- Plano atual: [PLAN.md](PLAN.md)
- Estratégia de testes: [TESTING.md](TESTING.md)
- Workflow detalhado: [WORKFLOW.md](WORKFLOW.md)
- README humano: [README.md](README.md)
- Paper de referência: *Architecting and Visualizing Deep Reinforcement Learning Models* (Neuwirth & Riley, MSOE) — em `docs/references/` e base das decisoes de modelo.
