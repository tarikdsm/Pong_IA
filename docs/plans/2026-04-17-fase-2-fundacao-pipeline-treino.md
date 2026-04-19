# PLAN.md

> Arquivo vivo. Uma feature ou tarefa por vez. Ao concluir, arquive em
> `docs/plans/<data>-<slug>.md` e limpe este arquivo para a proxima.
>
> **Regra:** o agente atualiza este arquivo antes de codar e marca progresso a
> cada passo. Se o plano nao existe ou esta desatualizado, a implementacao para.

---

## 0. Metadados

- **Feature / Tarefa:** Fase 2 - fundacao do pipeline de treino
- **Slug:** `fase-2-fundacao-pipeline-treino`
- **Autor humano:** _a definir_
- **Data de inicio:** 2026-04-17
- **Branch:** `feat/fase-2-fundacao-pipeline-treino`
- **Status:** em andamento

---

## 1. Contexto e Problema

- **Problema:** a Fase 2 ja tem `frame_stack`, `rollout`, `model.py` e policy
  estocastica, mas ainda falta a primeira atualizacao REINFORCE real dos pesos.
- **Quem sofre:** sem esse passo, o pipeline de treino continua sem gradiente
  aplicado, e o agente ainda nao aprende nada.
- **Por que agora:** o proximo degrau natural e conectar `log_probability`,
  retornos descontados e otimizador em uma iteracao pequena e testavel.

---

## 2. Critica da Ideia

- **Faz sentido tecnico?** Sim. Depois da policy, o trainer minimo ja pode
  existir sem pular direto para checkpoints, ONNX ou curriculo completo.
- **Ja existe algo no codigo que resolve isso?** Parcialmente: ja existem reward,
  retornos descontados, policy e modelo, mas o `reinforce.py` ainda nao executa
  uma iteracao fim a fim com backward e `optimizer.step()`.
- **Existe solucao mais simples?** Sim: treinar inline em um script ad hoc.
  Rejeitada porque espalharia logica de loss e coleta de gradiente fora do
  modulo de treino.
- **Riscos identificados:**
  - Normalizar retornos de episodio curto e zerar todo o gradiente sem perceber.
  - Duplicar a logica de rollout em vez de reaproveitar helpers existentes.
  - Fazer backward com tensores desconectados do grafo.
- **Suposicoes:**
  - Uma unica iteracao de treino ja e suficiente para validar o fluxo.
  - O trainer pode usar a mesma seed do rollout e o mesmo RNG `numpy`.
- **Alternativas rejeitadas:** pular direto para multiplos episodios por batch;
  adicionar persistencia agora; mover otimizacao para fora de `reinforce.py`.

Nenhum problema serio identificado; seguir.

---

## 3. Requisitos

### 3.1 Funcionais

- [x] Criar o pacote `training/` com `pyproject.toml`, estrutura `src/` e
  `tests/`.
- [x] Implementar `training/src/frame_stack.py` com stack logico de 5 frames
  binarios `60x80`.
- [x] A API deve expor o tensor atual em shape `5x60x80` e tambem uma visao
  achatada `24000`.
- [x] Frames faltantes no inicio do episodio devem ser preenchidos de forma
  deterministica.
- [x] Manter buffer rotativo em memoria com no maximo 10 snapshots de debug.
- [x] O buffer de debug deve descartar o mais antigo quando receber o 11o
  snapshot.
- [x] O contrato deve aceitar bitmaps gerados pelo motor Python sem depender do
  web runtime.
- [x] Implementar `training/src/rollout.py` para rodar episodios headless com o
  motor Python.
- [x] A politica da raquete direita deve receber observacao achatada em `24000`
  e retornar uma `Action`.
- [x] A raquete esquerda deve continuar controlada por `partially_tracking`.
- [x] O rollout deve registrar, por passo, observacao, acao, reward e estados
  antes e depois.
- [x] O reward base deve ser `+1` quando a direita pontua, `-1` quando a
  esquerda pontua e `0` nos demais passos.
- [x] O episodio deve terminar ao atingir `max_steps` ou `SCORE_TO_WIN`.
- [x] O rollout deve ser deterministico para a mesma seed e mesma politica.
- [x] Implementar `training/src/reinforce.py` com calculo de retornos
  descontados a partir de rewards escalares.
- [x] Expor helper para derivar retornos descontados diretamente de um
  `RolloutEpisode`.
- [x] Expor normalizacao segura de retornos para preparar o trainer futuro.
- [x] Validar `gamma` fora do intervalo `[0, 1]` com erro claro.
- [x] Declarar em `pyproject.toml` as dependencias Python necessarias para o
  slice de treino atual.
- [x] Instalar `uv` e sincronizar o ambiente Python do projeto.
- [x] Implementar `training/src/model.py` com MLP `24000 -> 200 -> 200 -> 100 -> 3`.
- [x] Validar shape de entrada do modelo com erro claro.
- [x] Expor logits e ativacoes intermediarias da politica para uso futuro em
  treino e visualizacao.
- [x] Implementar `training/src/policy.py` para converter observacao achatada em
  uma decisao de politica estocastica.
- [x] Mapear explicitamente indices dos logits para `Action`.
- [x] Expor helper que adapte o modelo ao callback exigido por `run_episode`.
- [x] Garantir que amostragem com mesma seed e mesmo modelo seja deterministica.
- [x] Implementar no `training/src/reinforce.py` uma iteracao REINFORCE minima
  que rode episodio, calcule loss e aplique `optimizer.step()`.
- [x] Expor metrica de saida com ao menos `loss`, `total_reward`,
  `episode_length` e `returns`.
- [x] Garantir que uma iteracao de treino altere pesos quando houver sinal de
  gradiente.

### 3.2 Nao-funcionais

- **Determinismo:** mesma seed e mesma politica produzem exatamente a mesma
  trajetoria.
- **Acoplamento:** reward e coleta de trajetoria ficam no pacote `training/`,
  nao no motor.
- **Clareza:** o contrato de rollout deve ser pequeno, puro e facil de evoluir
  para REINFORCE.
- **Estabilidade numerica:** normalizacao nao pode gerar `nan` em episodios
  degenerados.
- **Reprodutibilidade:** as dependencias necessarias para treino devem ficar
  declaradas no repositorio, nao apenas instaladas localmente.
- **Compatibilidade:** a policy precisa conversar com `rollout.py` sem mudar o
  contrato atual do callback.
- **Testabilidade:** a primeira iteracao de treino precisa ser exercitavel em um
  teste rapido, sem checkpoint nem I/O.

### 3.3 Fora do escopo

- Loop REINFORCE completo.
- Checkpoints.
- Exportacao ONNX.
- Inferencia no browser.

---

## 4. Criterios de Aceitacao

**Caminho feliz:**

- [x] Dado um `FrameStack` vazio, quando um bitmap valido e adicionado, entao o
  tensor atual possui shape `5x60x80`.
- [x] Dado 5 frames sequenciais validos, quando o tensor achatado e solicitado,
  entao seu comprimento e `24000`.
- [x] Dado mais de 10 snapshots de debug, quando o 11o e salvo, entao o buffer
  mantem apenas os 10 mais recentes.
- [x] Dado um rollout com `max_steps=3`, quando a politica da direita e
  executada, entao o episodio retorna 3 passos com observacoes achatadas em
  `24000`.
- [x] Dada a mesma seed e a mesma politica deterministica, quando dois rollouts
  sao executados, entao a trajetoria resultante e identica.
- [x] Dado um vetor de rewards e `gamma=0.5`, quando o desconto e calculado,
  entao cada posicao acumula corretamente as recompensas futuras ponderadas.
- [x] Dado um `RolloutEpisode`, quando os retornos sao derivados, entao o vetor
  resultante tem o mesmo comprimento do episodio.
- [x] Dado um batch com shape `(2, 24000)`, quando o modelo executa forward,
  entao os logits retornam shape `(2, 3)`.
- [x] Dado um forward valido, quando as ativacoes intermediarias sao pedidas,
  entao elas preservam as larguras `200`, `200` e `100`.
- [x] Dada uma observacao valida, quando a policy decide uma acao, entao ela
  retorna uma `Action` valida e probabilidades somando 1.
- [x] Dado o mesmo modelo e a mesma seed, quando a policy amostra uma sequencia
  de acoes, entao o resultado e identico.
- [x] Dado um modelo adaptado como callback, quando o `run_episode` o usa,
  entao o episodio roda sem wrapper manual extra.
- [x] Dado um conjunto simples de `log_probabilities` e `returns`, quando a loss
  REINFORCE e calculada, entao ela segue a formula esperada.
- [x] Dado um modelo e um otimizador, quando uma iteracao de treino roda,
  entao o resultado retorna metricas finitas e comprimento de episodio valido.
- [x] Dado um episodio com reward nao-trivial, quando uma iteracao de treino
  roda, entao ao menos um parametro do modelo muda.

**Casos de borda:**

- [x] Frames iniciais faltantes sao preenchidos deterministicamente.
- [x] Frame com shape invalido falha com erro claro.
- [x] Frame com valores fora de `{0, 1}` falha com erro claro.
- [x] Dado um estado inicial a um passo de a direita marcar, quando o rollout
  avanca, entao o reward do passo e `+1`.
- [x] Dado `max_steps <= 0`, quando o rollout e solicitado, entao ocorre erro
  claro.
- [x] Dado `gamma < 0` ou `gamma > 1`, quando o desconto e solicitado, entao
  ocorre erro claro.
- [x] Dado um vetor de retornos constantes, quando a normalizacao e solicitada,
  entao o resultado vira vetor de zeros em vez de `nan`.
- [x] Dado input com shape diferente de `24000`, quando o modelo roda forward,
  entao ocorre erro claro.
- [x] Dado ambiente sem dependencias de treino, quando a sincronizacao e feita,
  entao `torch` e `onnx` ficam importaveis.
- [x] Dada uma observacao com shape invalido, quando a policy decide acao,
  entao ocorre erro claro.
- [x] Dado numero de `log_probabilities` diferente do numero de retornos,
  quando a loss e calculada, entao ocorre erro claro.

---

## 5. Design Tecnico

### 5.1 Abordagem em uma frase

Conectar agora policy, retorno descontado e otimizador em uma iteracao
REINFORCE minima, mantendo o escopo curto e sem I/O adicional.

### 5.2 Arquivos que vao mudar

| Arquivo | Acao | Motivo |
|---|---|---|
| `PLAN.md` | atualizar | registrar o sexto slice da Fase 2 |
| `training/src/reinforce.py` | expandir | iteracao REINFORCE minima |
| `training/src/policy.py` | ajustar se preciso | compartilhar sampling com gradiente |
| `training/tests/test_reinforce.py` | expandir | TDD da loss e da iteracao |
| `training/src/__init__.py` | atualizar | exportar novos contratos |

### 5.3 Contratos / tipos novos

```python
@dataclass(frozen=True)
class ReinforceUpdateResult:
    loss: float
    total_reward: float
    episode_length: int
    returns: np.ndarray

def compute_reinforce_loss(
    log_probabilities: Sequence[torch.Tensor],
    returns: torch.Tensor,
) -> torch.Tensor: ...

def run_reinforce_update(
    network: PongPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    max_steps: int,
    gamma: float,
    initial_state: GameState | None = None,
) -> ReinforceUpdateResult: ...
```

---

## 6. Estrategia de Testes

- **Unitarios Python:** contrato de rollout, reward, termino, determinismo,
  desconto de rewards, normalizacao, shapes do modelo e policy estocastica.
- **Sem I/O real:** nada de checkpoints, PNG ou ONNX nesta etapa.
- **TDD obrigatorio:** testes vermelhos antes da implementacao.

---

## 7. Passos de Implementacao

- [x] **P1.** Arquivar o plano da Fase 1 e abrir este `PLAN.md`.
- [x] **P2.** Criar testes vermelhos para `FrameStack` e debug buffer.
- [x] **P3.** Criar o pacote `training/` com `pyproject.toml` e `src/__init__.py`.
- [x] **P4.** Implementar `training/src/frame_stack.py` ate os testes passarem.
- [x] **P5.** Rodar `ruff` e `pytest` no escopo Python relevante.
- [x] **P6.** Criar testes vermelhos para `training/src/rollout.py`.
- [x] **P7.** Implementar `training/src/rollout.py` com episodio, reward e
  termino deterministicos.
- [x] **P8.** Rodar `ruff` e `pytest` novamente no escopo Python.
- [x] **P9.** Criar testes vermelhos para `training/src/reinforce.py`.
- [x] **P10.** Implementar helpers de retornos descontados e normalizacao.
- [x] **P11.** Rodar `ruff`, `pytest` e checks finais do projeto.
- [x] **P12.** Declarar dependencias de treino em `pyproject.toml` e
  `training/pyproject.toml`.
- [x] **P13.** Instalar `uv` e sincronizar o ambiente Python.
- [x] **P14.** Criar testes vermelhos para `training/src/model.py`.
- [x] **P15.** Implementar `training/src/model.py` com forward validado.
- [x] **P16.** Rodar `ruff`, `pytest` e checks finais do projeto.
- [x] **P17.** Criar testes vermelhos para `training/src/policy.py`.
- [x] **P18.** Implementar a policy estocastica e o adaptador para rollout.
- [x] **P19.** Rodar `ruff`, `pytest` e checks finais do projeto.
- [x] **P20.** Criar testes vermelhos para a loss e a iteracao em
  `training/src/reinforce.py`.
- [x] **P21.** Implementar a primeira iteracao REINFORCE com backward e
  `optimizer.step()`.
- [x] **P22.** Rodar `ruff`, `pytest` e checks finais do projeto.

---

## 8. Rollout e Reversibilidade

- **Feature flag?** Nao.
- **Migracao de dados?** Nao.
- **Plano de rollback:** reverter a branch; nada externo e persistido nesta fase.

---

## 9. Checklist Final

- [x] Todos os criterios de aceitacao tem teste automatizado verde.
- [x] `uv run ruff check .` passa.
- [x] `uv run pytest` passa.
- [x] `PLAN.md` reflete o estado real da implementacao.

---

## 10. Log de Decisoes (append-only)

- **2026-04-17** - a Fase 2 sera iniciada pelo `frame_stack`, nao pelo loop de
  treino. Motivo: e o menor contrato estrutural com maior poder de desbloqueio.
- **2026-04-17** - o `pytest` da raiz passa a incluir `training/tests`. Motivo:
  a Fase 2 ja abriu um novo pacote Python que precisa entrar no caminho padrao
  de validacao.
- **2026-04-17** - o segundo slice da Fase 2 sera `rollout.py` antes de
  qualquer trainer. Motivo: consolidar trajetoria, reward e termino antes de
  introduzir PyTorch no loop.
- **2026-04-17** - o terceiro slice da Fase 2 sera o calculo de retornos em
  `reinforce.py` antes do trainer completo. Motivo: validar a matematica do
  algoritmo sem depender de PyTorch no ambiente atual.
- **2026-04-17** - o quarto slice da Fase 2 comeca pela sincronizacao de
  dependencias e pelo `model.py`. Motivo: o ambiente atual ainda nao consegue
  sustentar o proximo passo do treino.
- **2026-04-17** - o quinto slice da Fase 2 sera a policy estocastica antes do
  trainer REINFORCE. Motivo: fechar a fronteira entre logits do modelo e acao do
  jogo com determinismo explicito.
- **2026-04-17** - o sexto slice da Fase 2 sera uma iteracao REINFORCE minima.
  Motivo: validar o primeiro passo real de aprendizagem sem abrir escopo de
  scripts, checkpoints ou exportacao.
