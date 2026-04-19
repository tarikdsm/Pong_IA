# ARCHITECTURE.md

> Documenta **como o projeto está organizado** e **por quê**. Agentes devem lê-lo antes de introduzir novos módulos, padrões ou dependências.
>
> Se uma decisão aqui não faz mais sentido, **mude-a explicitamente** (seção 8) — não contorne em silêncio.

---

## 1. Princípios Arquiteturais

Em ordem de prioridade:

1. **Simples > Inteligente.** Prefira a solução óbvia que o próximo dev entende em 30 segundos.
2. **Paridade é contrato.** O motor Python e TS são a mesma máquina; divergência é bug.
3. **Determinismo sempre.** Mesmo input + mesmo seed = mesmo output, bit-a-bit. Sem `Math.random()` solto, sem `datetime.now()` no caminho quente.
4. **Boundaries explícitos.** Engine não conhece UI; UI não conhece otimizador; treino não conhece Canvas.
5. **Fail fast, fail loud.** Validar cedo, erros específicos, nunca silenciar exceção.
6. **Funções puras no núcleo.** Toda regra de física e renderização é pura. Efeitos colaterais só em `main.ts`, `train.py` e boundary de I/O.
7. **Stateless por padrão.** Estado só onde é inevitável — o `GameState` é passado adiante, nunca escondido em singleton.

---

## 2. Estilo Arquitetural

**Escolha:** **Modular Monolith com 3 pacotes** (`engine/`, `training/`, `web/`) unidos por um **contrato de paridade** sobre um `shared/config.json`.

**Motivo:** o projeto tem três modos operacionais distintos (simulação headless para treino, inferência em browser, jogo interativo) mas compartilha **uma única lógica de física e renderização**. Microserviços seriam overkill; monolito único mistura Python e TS. Três pacotes coordenados com contrato explícito equilibram os dois mundos.

### Camadas (por pacote)

```
engine/ (Python — canônico)
  pong_engine/
    config.py         ← gerado a partir de shared/config.json
    state.py          ← dataclass GameState (puro)
    physics.py        ← step(state, a_left, a_right, rng) → state' (puro)
    rendering.py      ← bitmap_from_state(state) → np.ndarray (puro)
    heuristics.py     ← partially_tracking(state) → action (puro)

training/ (Python)
  src/
    model.py          ← MLP PyTorch
    frame_stack.py    ← stack de 5 frames + buffer rotativo de debug (puro)
    rollout.py        ← USA engine/ para gerar episódios
    reinforce.py      ← otimização
    export_onnx.py    ← serializa modelo treinado

web/ (TypeScript)
  src/
    engine/           ← PORT espelhado de engine/pong_engine/ (paridade)
    ai/               ← carrega model.onnx e roda inferência
    viz/              ← visualização de frames, ativações e pesos
    ui/               ← scoreboard, controles e fluxo de rodada
    main.ts           ← game loop, orquestra tudo
```

**Regra de dependência:**

- `training/` importa de `engine/` (não o contrário).
- `web/src/ai` importa de `web/src/engine` (não o contrário).
- `web/src/ui` importa de `web/src/engine` e integra `web/src/ai` só no boundary de app.
- `engine/` e `web/src/engine` não importam nada além de stdlib/numpy/tipos primitivos TS.

---

## 3. Mapa de Módulos

| Pasta | Responsabilidade | Pode importar de | NÃO pode importar de |
|---|---|---|---|
| `engine/pong_engine` | Física, estado, rendering, heurística (Python puro) | stdlib, numpy | training, web, torch |
| `training/src` | Modelo, treino, exportação ONNX | engine, stdlib, torch, numpy, onnx | web |
| `web/src/engine` | Espelho TS de `engine/pong_engine` | stdlib TS | onnxruntime, DOM |
| `web/src/ai` | Frame stack e inferência ONNX | web/src/engine, onnxruntime-web | DOM |
| `web/src/viz` | Painel de visualização do modelo | web/src/ai, web/src/engine | lógica de treino |
| `web/src/ui` | Scoreboard, controles e round flow | web/src/engine | lógica de treino |
| `web/src/main.ts` | Bootstrap e game loop | todos acima | — |
| `shared/` | `config.json`, fixtures | — | — |
| `scripts/` | Geradores e verificadores de paridade | engine, shared | training, web |

---

## 4. Padrões Adotados

### 4.1 Injeção de Dependência

- RNG injetado em toda função que usa aleatoriedade. Python: `numpy.random.Generator` passado como argumento. TS: interface `{ next(): number }` implementada por PRNG determinístico (ex.: mulberry32).
- `physics.step()` não conhece RNG global; recebe-o.
- Testes substituem RNG por versão com seed fixa.

### 4.2 Tratamento de Erros

- **Erros esperados** (input inválido, estado inconsistente) → retorno tipado (Python: `Result`-like via tupla `(ok, value | error)`; TS: union `{ kind: 'ok', value } | { kind: 'err', error }`). Não lance.
- **Erros inesperados** (bug, invariante quebrado) → lance exceção; capture só no boundary (`main.ts`, `train.py`) para mapear em log/UI.
- Hierarquia de erros de domínio em `engine/pong_engine/errors.py` e `web/src/engine/errors.ts` (nomes equivalentes).

### 4.3 Validação de Input

- **Teclado/controles de UI:** normalizados em `web/src/ui/controls.ts` antes de chegar ao motor.
- **Modelo ONNX:** ao carregar, verificar shape de entrada/saída contra constantes esperadas de `config.ts`. Se divergente, falhar cedo com mensagem clara.
- **Config compartilhada:** `shared/config.json` validado no gerador (`scripts/gen_config.py`); valores fora de range rejeitados.

### 4.4 Logging e Observabilidade

- **Treino:** progresso humano-legível em stdout e histórico persistido dentro dos checkpoints.
- **Debug visual do treino:** materializar somente os bitmaps de debug pedidos para a rodada; nada no hot loop por padrão.
- **Web:** `console.log` só em desenvolvimento; em produção, nenhum serviço externo é configurado por padrão.
- **Nunca logar:** paths absolutos do usuário, tokens, dados que identifiquem a máquina.

### 4.5 Motor Espelhado — regras de paridade Python↔TS

> Seção substitui "Acesso a Banco" do template padrão, pois não há DB.

- **Fonte única de verdade:** lógica canônica está em Python. Quando uma regra muda, muda primeiro em `engine/`, depois em `web/src/engine/`, no **mesmo PR**.
- **Nomes e assinaturas:** funções equivalentes têm o mesmo nome (case-ajustado) e a mesma ordem de argumentos. `physics.step(state, a_left, a_right, rng)` em ambos.
- **Tipos numéricos:** usar `float64` explicitamente em Python; `number` em TS opera em `float64` por padrão. Valores inteiros (coordenadas de pixel) são `int` / `number & integer`.
- **Gerador de fixtures:** `scripts/gen_fixture.py` joga uma partida inteira com seed fixa e grava `(seed, ações, estados por step)` em `shared/fixtures/*.json`.
- **Verificador:** `scripts/parity_check.py` carrega fixture, re-executa em Python, compara com arquivo. A suite TS em `web/tests/parity.test.ts` carrega a mesma fixture e re-executa em TS. Tolerância float: `1e-6`.
- **Falha de paridade é CRÍTICA** — não fazer merge até resolver.

### 4.6 Inferência ONNX no Browser

> Seção substitui "HTTP / Serviços Externos" do template padrão.

- Modelo exportado de PyTorch para ONNX com opset estável (17+). Salva em `web/public/model.onnx` (servido estaticamente pelo Vite).
- Carregamento: `InferenceSession.create('/model.onnx', { executionProviders: ['wasm'] })` na inicialização. Fallback: `'webgl'` em ambientes que suportam.
- **Saída atual do modelo:** logits da política + ativações das três camadas ocultas.
- **Artefatos de visualização:** `training/scripts/export.py` também gera `web/public/model-viz.json` e `web/public/model-first-layer.uint8.bin` com os pesos quantizados da primeira camada para o painel do browser.
- **Cadência de inferência:** rodar forward a cada N ticks do game loop (default atual `30`). A ação do modelo é mantida entre inferências.

### 4.7 Configuração

- `shared/config.json` é a fonte única. Contém dimensões da arena, velocidades iniciais, fator de aceleração, tamanho da raquete, bitmap HxW, seed default, taxa de inferência.
- `scripts/gen_config.py` lê o JSON e escreve `engine/pong_engine/config.py` e `web/src/engine/config.ts` com os mesmos valores; arquivos gerados têm cabeçalho `# AUTO-GERADO — não edite manualmente`.
- Zero leitura de `os.environ` ou `process.env` no caminho de jogo/treino. Variáveis de ambiente só em scripts de entrada.

---

## 5. Convenções de Código Reforçadas

Ver também `AGENTS.md` §6. Decisões com **peso arquitetural**:

- **Funções puras** em `engine/` e `web/src/engine/`. Efeitos apenas em `main.ts`/`train.py`.
- **Imutabilidade** preferida. `GameState` é dataclass/interface read-only; `step()` retorna um novo estado.
- **Nenhum singleton global mutável.** RNG e config são passados.
- **Nenhum import circular.**
- **`GameState` tem shape congelado.** Adicionar campo é ADR (seção 8) porque quebra compatibilidade de fixtures e ONNX.

---

## 6. Segurança (princípios)

- **Zero trust em arquivos de modelo.** Validar shape, dtypes e faixa de saída ao carregar.
- **Secrets:** não há, por ora. Se aparecer (ex.: upload a servidor externo), passa por `config` e env, nunca em código.
- **Dependências** atualizadas; auditoria via `uv run pip-audit` e `pnpm audit` em CI.
- **CSP no `index.html`** restrito a `self` + blobs (ONNX pode usar `wasm-eval`).

---

## 7. Performance e Escalabilidade (guias)

- **Treino:** CPU é suficiente para MLP denso de ~5M parâmetros; GPU é bônus, não requisito. Meça antes de otimizar.
- **Browser:** game loop em 60 fps no Canvas; inferência em 10–20 Hz (ver 4.6). Se frame demorar > 16ms, perfil com DevTools antes de otimizar.
- **Memória:** frame stack é ring buffer de 5 bitmaps (~24 KB cada em `Uint8Array`). Zero alocações por frame no caminho quente — reaproveitar buffers.
- **Persistência de debug:** snapshots de depuração não podem crescer sem limite; o contrato é buffer máximo de 10 frames salvos.
- **Evitar:** `JSON.parse`/`JSON.stringify` no game loop. Clonagens desnecessárias do `GameState`.

---

## 8. Log de Decisões Arquiteturais (ADR-lite)

> Formato append-only. Nunca edite uma decisão passada; adicione nova que a substitua.

### ADR-0001: Motor Espelhado com Testes de Paridade

- **Data:** 2026-04-17
- **Status:** Aceito
- **Contexto:** o treino precisa rodar headless e rápido em Python (PyTorch); o browser precisa rodar o mesmo jogo em tempo real. Opções: (a) Python compilado para WASM via Pyodide, (b) motor em Rust/WASM com bindings, (c) motor duplicado Python+TS com contrato de paridade.
- **Decisão:** (c). Duplicar código é aceitável porque a superfície é pequena (física determinística simples) e o benefício (debug trivial, sem toolchain exótica) é grande.
- **Alternativas rejeitadas:** (a) Pyodide é lento em loop apertado e complica deploy; (b) Rust adiciona terceira linguagem e barreira de contribuição para um projeto educacional.
- **Consequências:** toda mudança no motor obriga mudança nos dois lados e no CI. Mitigado por `scripts/parity_check.py` e `web/tests/parity.test.ts` que rejeitam drift.

### ADR-0002: MLP Denso em vez de CNN

- **Data:** 2026-04-17
- **Status:** Aceito
- **Contexto:** objetivo primário é **visualização de padrões espaciais** dentro da rede, não score máximo. CNNs aprendem filtros locais úteis, mas escondem o "mapa do jogo" atrás de camadas convolucionais de features.
- **Decisão:** MLP denso com input `flatten(5, 60, 80) = 24000` → 200 → 200 → 100 → 3. Pesos da 1ª camada, remodelados em `5×60×80`, são diretamente interpretáveis como "5 imagens de peso" por neurônio.
- **Alternativas rejeitadas:** CNN (obscurece pesos); RNN/LSTM (complica exportação ONNX e visualização temporal sem benefício claro).
- **Consequências:** ~4.8M parâmetros na 1ª camada oculta — aceitável para um MLP educacional; treino mais lento que CNN equivalente, mas ainda tratável em CPU.

### ADR-0003: 5 Frames Empilhados em vez de Diferença

- **Data:** 2026-04-17
- **Status:** Aceito
- **Contexto:** o paper de referência (Neuwirth & Riley) usa diferença entre dois frames como input para capturar velocidade. Isso tem dois problemas documentados no próprio paper: §4.1 perde bola parada entre frames idênticos; §4.2 raquetes paradas ficam invisíveis.
- **Decisão:** empilhar 5 bitmaps sequenciais brutos (valores em `{0, 1}`) e achatar — o modelo tem todas as informações para inferir posição, velocidade e aceleração.
- **Alternativas rejeitadas:** diferença única (perde info), 2-3 frames (insuficiente para aceleração), 10+ frames (RAM/compute).
- **Consequências:** input maior (24k vs. 4.8k pixels), primeira camada maior, mais parâmetros para visualizar. Aceitável para o objetivo educacional.

### ADR-0004: Entrada do Modelo como Um Único Tensor Contíguo de 5 Frames

- **Data:** 2026-04-17
- **Status:** Aceito
- **Contexto:** a IA precisa consumir 5 frames sequenciais, mas representá-los como 5 arquivos ou 5 imagens independentes aumenta atrito no pipeline de treino e debug.
- **Decisão:** o contrato de entrada do modelo será um único tensor/array contínuo com shape lógico `5×60×80`, preservando a separação temporal por frame. Para o MLP, esse tensor é achatado apenas no boundary de entrada para `24000`.
- **Alternativas rejeitadas:** concatenar frames em um mosaico 2D ad hoc (piora legibilidade sem ganho claro); salvar cada frame como amostra separada (quebra contexto temporal).
- **Consequências:** o pipeline de treino opera com um único objeto por amostra; visualização e debug continuam conseguindo remontar os 5 frames originais sem heurística extra.

### ADR-0005: Currículo de Treino da Aceleração da Bola

- **Data:** 2026-04-17
- **Status:** Aceito
- **Contexto:** iniciar o treino já com aceleração alta aumenta a dificuldade da política antes que o agente aprenda o básico de rastrear a bola e devolver rebatidas consistentes.
- **Decisão:** os primeiros ciclos de treino devem começar com `acceleration_factor = 1.00` e evoluir em estágios até `1.05`, preferencialmente na sequência `1.00 → 1.01 → 1.02 → 1.03 → 1.04 → 1.05`.
- **Alternativas rejeitadas:** treinar sempre em `1.05` desde o início (aprendizado mais instável); manter `1.00` para sempre (modelo subtreinado para o cenário final).
- **Consequências:** o pipeline de treino precisa aceitar agenda explícita de aceleração e registrar em métricas qual estágio produziu cada checkpoint.

---

## 9. O Que Este Projeto Deliberadamente NÃO Faz

- Não suporta multi-jogador em rede.
- Não persiste partidas, replays ou rankings.
- Não tem autenticação nem contas de usuário.
- Não suporta controles touch para mobile; o produto atual foca em desktop.
- Não treina no browser (treino é exclusivo em Python; browser só infere).
- Não ajusta hiperparâmetros via UI — edite `shared/config.json` ou flags de `train.py`.
- Não implementa frameworks RL genéricos (Stable-Baselines, RLlib) — REINFORCE manual é parte do valor educacional.
- Não usa GPU obrigatoriamente; CPU deve completar o treino em tempo razoável.

Isso evita que um agente "ajude" adicionando complexidade fora do escopo.
