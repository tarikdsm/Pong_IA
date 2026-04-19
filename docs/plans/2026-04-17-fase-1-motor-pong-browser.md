# PLAN.md

> Arquivo vivo. Uma feature/tarefa por vez. Ao concluir, arquive em `docs/plans/<data>-<slug>.md` e limpe este arquivo para a próxima.
>
> **Regra:** o agente atualiza este arquivo antes de codar e marca progresso a cada passo. Se o plano não existe ou está desatualizado, a implementação para.

---

## 0. Metadados

- **Feature / Tarefa:** Fase 1 — MVP do motor Pong no browser (sem modelo ainda)
- **Slug:** `fase-1-motor-pong-browser`
- **Autor humano:** _a definir_
- **Data de início:** 2026-04-17
- **Branch:** `feat/fase-1-motor-pong-browser`
- **Status:** ✅ Implementado (smoke manual pendente)

---

## 1. Contexto e Problema

- **Problema:** não existe ainda nada além de templates — é preciso um motor de Pong funcional e determinístico antes de treinar qualquer modelo.
- **Quem sofre:** as fases seguintes (treino, inferência, visualização) dependem de um motor estável, replicado em Python e TypeScript, com paridade verificada.
- **Por que agora:** é a base do projeto. Qualquer bug de física aqui se propaga para o treino e para a visualização.

---

## 2. Crítica da Ideia

- **Faz sentido técnico?** Sim. A estratégia "motor canônico em Python, port em TS com paridade" é conservadora e bem documentada (ADR-0001 em `ARCHITECTURE.md`).
- **Já existe algo no código que resolve isso?** Não; o projeto parte do zero. O paper de referência (em `temp/`) traz detalhes valiosos sobre regras de Pong e deficiências conhecidas (§4 do paper: perda de informação em frames, raquetes paradas invisíveis).
- **Existe solução mais simples?** Sim — rodar **sem** paridade e refazer o motor direto em TS. Descartada porque o treino fica 10x mais lento em Node via TF.js e perde-se a vantagem de numpy no rollout.
- **Riscos identificados:**
  - Drift entre motor Python e TS (mitigado por fixtures golden + CI).
  - Determinismo quebrar sutilmente por diferença de aritmética float (mitigado por testar em valores inteiros quando possível; tolerância `1e-6` em floats).
  - Aceleração multiplicativa da bola fazer a física divergir em velocidades altas (mitigado com velocidade máxima clampada e teste de caso limite).
- **Suposições:**
  - Bitmap 60×80 é suficiente para representar raquete e bola sem ambiguidade (validar em §4).
  - 60 fps de game loop é sustentável em hardware comum (validar manualmente).
- **Alternativas rejeitadas:** Pyodide para rodar Python no browser; motor em Rust/WASM. Ver ADR-0001.

Nenhum problema sério identificado; seguir.

---

## 3. Requisitos

### 3.1 Funcionais

- [x] Motor Pong em Python (`engine/pong_engine/`) com física determinística: movimento da bola, colisão com paredes e raquetes, gol nas bordas esquerda/direita.
- [x] Aceleração progressiva: a cada rebatida a velocidade da bola é multiplicada por `acceleration_factor` (default 1.03), limitada por `max_ball_speed`.
- [x] Heurística "partially-tracking" na raquete esquerda: segue a bola **apenas quando ela se move em direção à raquete**; caso contrário, move aleatoriamente (RNG injetado).
- [x] `bitmap_from_state(state)` produz array binário (0/1) de shape `(H, W)` determinístico.
- [x] Port TypeScript em `web/src/engine/` com as mesmas funções, mesma assinatura, mesmo comportamento bit-a-bit.
- [x] Game loop no browser em 60 fps usando `requestAnimationFrame`; Canvas desenha arena, raquetes, bola, placar.
- [x] Controles de teclado: setas ↑/↓ (ou W/S) movem a raquete direita.
- [x] Scoreboard visível; partida termina em 21 pontos.
- [x] `shared/config.json` é a única fonte de constantes físicas; `scripts/gen_config.py` gera os arquivos Python e TS correspondentes.
- [x] `scripts/parity_check.py` + `web/tests/parity.test.ts` verificam paridade em 3+ fixtures golden.

### 3.2 Não-funcionais

- **Performance:** game loop mantém 60 fps em máquina comum (laptop 2020+). Um step do motor < 0.5 ms em Python, < 0.1 ms em TS.
- **Determinismo:** mesma seed + mesma sequência de ações → mesmo estado final, em ambas as linguagens.
- **Observabilidade:** console do browser mostra FPS; sem logs espalhados.
- **Compatibilidade:** Chrome/Firefox/Edge recentes. Sem suporte a IE.

### 3.3 Fora do escopo

- Modelo de IA — Fase 2.
- Inferência ONNX — Fase 3.
- Visualização de ativações — Fase 4.
- Mobile/touch.
- Sons.

---

## 4. Critérios de Aceitação

**Caminho feliz:**

- [x] Dado o estado inicial canônico, quando 1000 steps são executados com ações `[NONE, NONE, ...]` em ambos os motores, então os estados finais são iguais (paridade).
- [x] Dado uma partida simulada Python com seed 42 e ações geradas pela heurística, quando a mesma fixture é re-executada em TS, então placar e posições finais batem.
- [x] Dado a bola batendo em uma raquete, quando o passo é executado, então `ball_speed` é multiplicado por `acceleration_factor` (clampado em `max_ball_speed`).
- [x] No browser, apertar ↑ move a raquete direita para cima; soltar para-a.
- [x] Partida chega a 21 pontos e o estado reseta.

**Casos de borda:**

- [x] Bola saindo pela lateral esquerda: incrementa placar direito, reseta bola no centro com velocidade base.
- [x] Bola saindo pela lateral direita: incrementa placar esquerdo, reseta bola no centro.
- [x] Bola atingindo canto superior/inferior enquanto também cruza raquete: só uma reflexão por step (definir ordem: parede primeiro, depois raquete).
- [x] Velocidade da bola atingindo `max_ball_speed`: não aumenta mais, mesmo com novas rebatidas.
- [x] Raquete direita tentando sair do topo: clamp em `y = 0`.
- [x] Raquete esquerda sem RNG: `physics.step` falha com erro claro (não cai em `None.next()`).

**Falhas esperadas:**

- [x] Modelo ONNX ausente: carregamento web deve ser graceful (Fase 1 não usa modelo; o botão "usar modelo" só aparece na Fase 3).
- [x] `shared/config.json` inválido (valores negativos, formatos errados): `gen_config.py` falha com mensagem específica, sem gerar arquivos parciais.

---

## 5. Design Técnico

### 5.1 Abordagem em uma frase

Construir o motor canônico em Python com testes unitários cobrindo cada regra, portar para TS espelhando nomes e assinaturas, validar paridade com fixtures geradas, e conectar ao Canvas no browser.

### 5.2 Arquivos que vão mudar

| Arquivo | Ação | Motivo |
|---|---|---|
| `shared/config.json` | criar | fonte única de constantes |
| `scripts/gen_config.py` | criar | derivação Python+TS |
| `scripts/gen_fixture.py` | criar | gera fixtures golden |
| `scripts/parity_check.py` | criar | verifica paridade Python↔TS |
| `engine/pyproject.toml` | criar | pacote Python |
| `engine/pong_engine/{__init__,config,state,physics,rendering,heuristics,errors}.py` | criar | motor canônico |
| `engine/tests/test_{physics,rendering,heuristics,parity}.py` | criar | cobertura do motor |
| `web/package.json`, `tsconfig.json`, `vite.config.ts`, `index.html` | criar | projeto Vite |
| `web/src/main.ts` | criar | game loop |
| `web/src/engine/{state,physics,rendering,bitmap,heuristics,config,errors}.ts` | criar | port TS |
| `web/src/ui/{scoreboard,controls}.ts` | criar | UI mínima |
| `web/tests/{engine,parity}.test.ts` | criar | cobertura TS |
| `shared/fixtures/*.json` | gerar | golden tests |
| `.gitignore` | criar (já existe via setup inicial) | ignorar `node_modules`, `.venv`, `checkpoints/` |

### 5.3 Contratos / tipos novos

```python
# engine/pong_engine/state.py
@dataclass(frozen=True)
class GameState:
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    ball_speed: float  # módulo da velocidade; usado para aceleração
    paddle_left_y: int
    paddle_right_y: int
    score_left: int
    score_right: int
    tick: int

Action = Literal["up", "down", "none"]

def step(state: GameState, a_left: Action, a_right: Action, rng: np.random.Generator) -> GameState: ...
def bitmap_from_state(state: GameState) -> np.ndarray: ...  # shape (H, W), dtype=uint8, values in {0, 1}
def partially_tracking(state: GameState, rng: np.random.Generator) -> Action: ...
```

```ts
// web/src/engine/state.ts
export interface GameState {
  readonly ballX: number; readonly ballY: number;
  readonly ballVx: number; readonly ballVy: number;
  readonly ballSpeed: number;
  readonly paddleLeftY: number; readonly paddleRightY: number;
  readonly scoreLeft: number; readonly scoreRight: number;
  readonly tick: number;
}
export type Action = 'up' | 'down' | 'none';
export interface Rng { next(): number; } // uniforme [0, 1)

export function step(state: GameState, aLeft: Action, aRight: Action, rng: Rng): GameState;
export function bitmapFromState(state: GameState): Uint8Array; // length H*W, values in {0, 1}
export function partiallyTracking(state: GameState, rng: Rng): Action;
```

### 5.4 Fluxo de dados

```
shared/config.json ──▶ gen_config.py ──▶ engine/.../config.py
                                      └──▶ web/src/engine/config.ts

engine/pong_engine ─── gen_fixture.py ──▶ shared/fixtures/*.json
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
           parity_check.py (Py)                    web/tests/parity.test.ts (TS)
                     │                                         │
                     └────────── compara estados ──────────────┘

web/src/main.ts ─ requestAnimationFrame ─▶ physics.step ─▶ rendering.draw ─▶ canvas
                                           ▲
                                           ├─ controls.ts (teclado)
                                           └─ heuristics.ts (raquete esquerda)
```

### 5.5 Dependências novas

- Python: `numpy` (array, RNG), `pytest`, `ruff`. Justificativa: padrão; zero surpresa.
- TS: `typescript`, `vite`, `vitest`, `@types/node`, `eslint`, `prettier`. Justificativa: ferramental mínimo do ecossistema.

Sem dependências exóticas.

---

## 6. Estratégia de Testes

Ver `TESTING.md`. Para esta fase:

- **Unitários Python:** física (movimento, colisão, aceleração, gol), rendering (bitmap shape/valores), heurística (casos bola-indo/bola-vindo).
- **Unitários TS:** mesmos casos, mesmos valores esperados.
- **Integração / Paridade:** 3 fixtures (partida curta com bola lenta, partida longa com aceleração, partida forçando bola em `max_ball_speed`).
- **Manual:** abrir `pnpm dev`, jogar, confirmar responsividade e visual.
- **Regressão:** cada bug achado vira fixture.

---

## 7. Passos de Implementação

Cada passo é 1 commit, TDD. Máximo 3 tentativas de auto-correção por passo antes de escalar.

- [x] **P1.** `shared/config.json` + `scripts/gen_config.py` + teste que roda gerador e verifica conteúdo em Python e TS.
- [x] **P2.** `engine/pong_engine/state.py` + `errors.py` + teste de construção/invariantes.
- [x] **P3.** `engine/pong_engine/physics.py` — movimento simples (sem colisão) + teste.
- [x] **P4.** `physics.py` — colisão com paredes horizontais + teste.
- [x] **P5.** `physics.py` — colisão com raquetes + aceleração + teste.
- [x] **P6.** `physics.py` — gol (bola sai lateral) + reset + teste.
- [x] **P7.** `engine/pong_engine/heuristics.py` partially-tracking + teste com RNG seedado.
- [x] **P8.** `engine/pong_engine/rendering.py` bitmap_from_state + teste com fixture de pixels esperados.
- [x] **P9.** `scripts/gen_fixture.py` — gera 3 fixtures JSON golden.
- [x] **P10.** `scripts/parity_check.py` — re-executa fixtures em Python e valida (sanidade).
- [x] **P11.** Port TS `web/src/engine/config.ts` (gerado), `state.ts`, `errors.ts` + testes vitest espelhando P2.
- [x] **P12.** Port TS `physics.ts` (espelha P3–P6) + testes.
- [x] **P13.** Port TS `heuristics.ts` + teste com PRNG determinístico (mulberry32 seedado).
- [x] **P14.** Port TS `bitmap.ts` + teste comparando com fixture Python.
- [x] **P15.** `web/tests/parity.test.ts` — carrega fixtures e valida paridade step-a-step.
- [x] **P16.** `web/src/ui/controls.ts` (teclado) + `scoreboard.ts` + testes.
- [x] **P17.** `web/src/engine/rendering.ts` (desenho em Canvas, não confundir com bitmap).
- [x] **P18.** `web/src/main.ts` — game loop, bootstrap, conecta tudo.
- [~] **P19.** Smoke manual: `pnpm dev`, jogar 2 partidas, validar 60 fps e aceleração visível.
- [x] **P20.** Rodar `scripts/parity_check.py` e `pnpm --filter web test` — tudo verde.

---

## 8. Rollout e Reversibilidade

- **Feature flag?** Não; é o MVP, não substitui nada em produção.
- **Migração de dados?** Não.
- **Plano de rollback:** reverter branch; nenhum efeito externo.
- **Impacto em usuários existentes:** nenhum (não há usuários ainda).

---

## 9. Checklist Final

- [x] Todos os critérios de aceitação (§4) têm teste automatizado verde.
- [ ] `uv run ruff check . && uv run pytest` passa.
- [x] `pnpm --filter web typecheck && pnpm --filter web lint && pnpm --filter web test` passa.
- [x] `python scripts/parity_check.py` passa em 3 fixtures.
- [ ] Smoke manual no browser passou (60 fps, aceleração visível, controles responsivos).
- [ ] Cobertura não caiu em relação ao estado anterior.
- [ ] Nenhum `TODO`/`FIXME` novo sem issue vinculada.
- [ ] Sem segredos, paths absolutos, ou dados pessoais hardcoded.
- [ ] Commits seguem Conventional Commits, um por ideia.
- [x] README atualizado com o quickstart funcionando.
- [ ] Este `PLAN.md` arquivado em `docs/plans/2026-04-17-fase-1-motor-pong-browser.md`.

---

## 10. Log de Decisões (append-only)

- **2026-04-17** — aceleração multiplicativa aplicada **apenas na rebatida em raquetes** (não em paredes). Motivo: mantém o ritmo do jogo previsível; rebatidas em paredes são frequentes demais e saturariam rápido.
- **2026-04-17** — velocidade máxima `max_ball_speed` definida em `shared/config.json`; valor inicial será ajustado em smoke test manual (§P19) até ficar "desafiador mas jogável".
- **2026-04-17** — `acceleration_factor` reduzido de `1.03` para `1.003`. Motivo: a taxa anterior acelerava rápido demais para o ritmo desejado do MVP.
- **2026-04-17** — saques e resets da bola passam a usar vetor aleatório com componente vertical não nula dentro de um cone controlado. Motivo: variar trajetórias sem introduzir lançamentos quase verticais ou triviais demais.
