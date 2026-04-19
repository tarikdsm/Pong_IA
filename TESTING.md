# TESTING.md

> Contrato de testes do projeto. Vale para humanos e agentes.
>
> **Regra de ouro:** uma funcionalidade só é considerada **pronta** quando existe teste automatizado verde que a cobre. Sem teste, não está pronta — mesmo que funcione manualmente.

---

## 1. Filosofia

- **Testes são documentação executável.** Um teste bem escrito explica *como usar* o código.
- **TDD é o padrão**, não a exceção. Red → Green → Refactor.
- **Teste comportamento, não implementação.** Se refatoração interna quebra teste, o teste está ruim.
- **Teste rápido roda sempre.** Teste lento afasta quem deveria rodá-lo.
- **Cobertura é piso, não teto.** 80% ruim vale menos que 60% bom.
- **Determinismo é pré-requisito.** Todo teste usa seed fixa; nenhum teste depende de `datetime.now()`, ordem de iteração de dict, ou `Math.random()` não-seedado.

---

## 2. Pirâmide de Testes

```
        ▲
       /E2E\        poucos (smoke manual), fluxo do browser ponta a ponta
      /─────\
     / Parid.\     camada própria: paridade Python↔TS (ver §13)
    /─────────\
   /  Integ.  \    integração engine↔treino, engine↔UI
  /─────────────\
 /  Unitários   \  muitos, rápidos, um comportamento por vez
/─────────────────\
```

Proporção-alvo: **65% unitários / 20% paridade / 12% integração / 3% E2E**. Paridade é categoria explícita devido ao motor duplicado.

---

## 3. Tipos de Teste

### 3.1 Unitários

**O que são:** testam **uma unidade pequena** (função, método, classe) **em isolamento**.

**Onde ficam:**
- Python: `engine/tests/`, `training/tests/` espelhando `src/`.
- TS: `web/tests/` ou `*.test.ts` ao lado do arquivo.

**Características:**

- Milissegundos por teste.
- **Sem I/O:** sem rede, sem disco, sem DB, sem clock real.
- Dependências externas são **mockadas/stubadas** via DI. RNG é injetado com seed.

**Priorize** para:

- Física do motor (`step`, colisões, aceleração).
- Rendering determinístico (`bitmap_from_state`).
- Frame stack, discount rewards, cálculo de gradiente.
- Validação de shapes do modelo.

### 3.2 Integração

Validam que **dois ou mais módulos reais** conversam corretamente.

Cobrem:

- `engine/` + `training/rollout.py`: um episódio completo com heurística contra modelo random.
- `engine/` + `training/reinforce.py`: uma iteração de treino não explode.
- `web/src/engine/` + `web/src/ui/controls.ts`: uma tecla pressionada move a raquete.
- `training/export_onnx.py` + `onnxruntime-web`: modelo exportado carrega e roda inferência com shapes corretos.

**Regras:**

- Zero dependências externas (sem servidor, sem DB).
- ONNX de teste é gerado on-the-fly com pesos aleatórios em setup.

### 3.3 End-to-End (E2E)

Simulam o **usuário real** no browser.

**Use somente para:**

- Smoke: abrir `localhost`, jogar 30 segundos, não crashar.
- Controles respondem (tecla → movimento).
- Aceleração da bola perceptível ao longo de uma partida.

**Ferramentas:** Playwright continua opcional; smoke manual ainda é aceitável para a partida completa no browser.

### 3.4 Contrato / ONNX

- O teste de export verifica que o ONNX expõe `logits`, `hidden_one`, `hidden_two` e `hidden_three` com as shapes esperadas.
- Os testes de browser validam que `web/src/ai/inference.ts` consome essas saídas e que `web/src/viz/model_viz.ts` continua funcional mesmo se o artefato auxiliar de pesos não estiver disponível.

### 3.5 Regressão

- Bug reportado → **primeiro** escreva o teste que reproduz o bug (deve falhar).
- **Depois** corrija o código até o teste passar.
- Teste permanece no suite para sempre.
- Fixtures de paridade funcionam como regressão cumulativa do motor.

---

## 4. Regras de Escrita

### 4.1 Estrutura AAA

Todo teste segue **Arrange → Act → Assert**:

```python
def test_bola_acelera_ao_rebater_na_raquete_direita():
    # Arrange
    cfg = load_test_config()
    state = GameState(ball_x=cfg.width - 2, ball_y=cfg.height // 2,
                      ball_vx=1.0, ball_vy=0.0, ball_speed=1.0,
                      paddle_left_y=0, paddle_right_y=cfg.height // 2,
                      score_left=0, score_right=0, tick=0)
    rng = np.random.default_rng(42)

    # Act
    next_state = step(state, "none", "none", rng)

    # Assert
    assert next_state.ball_speed == pytest.approx(1.0 * cfg.acceleration_factor)
```

### 4.2 Nomeação

- Descreva **comportamento esperado**, não implementação.
- ✅ `test_bola_acelera_ao_rebater_na_raquete_direita`
- ❌ `test_step_case_4`

### 4.3 Um comportamento por teste

- Um `test`/`it` valida uma afirmação.
- Muitos `assert`/`expect` só se forem facetas da mesma afirmação.

### 4.4 Independência

- Testes não dependem de ordem.
- Cada teste cria seu próprio estado.
- Use fixtures (`pytest` / `beforeEach`) para setup, nunca globais mutáveis.

### 4.5 Dados de teste

- **Builders / factories** em vez de literais repetidos:

  ```python
  state = a_state().with_ball_at(10, 20).with_ball_velocity(1, 0).build()
  ```

- Dados mínimos necessários. Nada além.

### 4.6 Assertivas específicas

- `assert x == 5` > `assert x`
- Compare float com tolerância (`pytest.approx`, `expect(x).toBeCloseTo`).
- Mensagens de erro esperadas também são verificadas.

---

## 5. Mocks, Stubs e Fakes

- **Fakes > Mocks.** Implementações em memória são mais robustas que `MagicMock`.
- Mocke apenas **dependências externas**; nunca o próprio código sob teste.
- Não mocke o que você não possui — envolva em adaptador e mocke o adaptador.
- **Zero mocks no motor.** Estado real, física real.
- RNG sempre seedado, nunca mockado — é função pura com seed, não serviço.

---

## 6. O Que Testar

### 6.1 Caminho feliz

Entrada válida típica produz saída esperada.

### 6.2 Casos de borda

- Bola em `x = 0` ou `x = width`.
- Raquete no topo/fundo (`y = 0` / `y = height - paddle_height`).
- Velocidade da bola no limite `max_ball_speed`.
- Frame stack com menos de 5 frames (início de episódio).
- Ações recebidas inválidas.
- Seeds diferentes produzem trajetórias diferentes; seeds iguais produzem iguais.

### 6.3 Falhas esperadas

- `config.json` com valores negativos → gerador falha com mensagem clara.
- ONNX com shape errado → loader falha cedo.
- Ação inválida → erro de tipo no boundary, não propaga.

### 6.4 Não teste

- `numpy`/`torch`/`onnxruntime` em si.
- Getters/setters triviais.
- Código gerado (`config.py`/`config.ts`) — testa-se o gerador, não o artefato.

---

## 7. Cobertura

- **`engine/`:** 85%+ linhas, 80%+ branches — é o núcleo determinístico e precisa ser blindado.
- **`training/`:** 70%+ — partes numéricas (gradiente, desconto) precisam de teste; loops de treino em si podem ser cobertos por smoke de integração.
- **`web/`:** 75%+ — engine TS espelha a cobertura do Python; `ui/` e integração da app devem manter smoke robusto.
- Cobertura medida em CI; queda > 2% em relação à main **bloqueia merge**.
- **Cobertura não é objetivo, é termômetro.**

---

## 8. Ciclo TDD (Red → Green → Refactor)

1. **Red:** escreva o menor teste possível que falha pelo motivo certo.
2. **Green:** escreva o **mínimo** de código para passar.
3. **Refactor:** melhore estrutura com os testes te protegendo.
4. Repita.

**Regras:**

- Nunca escreva código de produção sem um teste vermelho exigindo.
- Nunca escreva mais teste do que o suficiente para falhar.
- Nunca escreva mais produção do que o suficiente para o teste passar.

---

## 9. Execução

| Quando | O que |
|---|---|
| Durante dev (engine Python) | `uv run pytest engine/tests -k <padrão> --watch` (com `pytest-watcher`) |
| Durante dev (web) | `pnpm --filter web test --watch` |
| Antes de commit | `uv run pytest` (tudo) + `pnpm --filter web test` + paridade |
| Antes de push | unit + integração + paridade + lint + typecheck |
| CI em PR | tudo acima + cobertura |
| CI em main | tudo acima + smoke E2E (quando disponível em Fase 3+) |

**Comandos canônicos:**

```bash
# Python (da raiz)
uv run ruff check .
uv run pytest
uv run pytest --cov=engine --cov-report=term-missing

# Web
pnpm --filter web lint
pnpm --filter web typecheck
pnpm --filter web test
pnpm --filter web test --coverage

# Paridade (gate crítico)
python scripts/parity_check.py
```

**Regra:** commit com teste vermelho é proibido. CI bloqueia merge.

---

## 10. Testes em Código Gerado por IA

> Seção específica para Vibe Coding.

- **Nunca aceite "passou nos meus testes" sem rodar você também.**
- Se o agente escreveu código **e** teste, desconfie: pode ter moldado o teste para o código. Peça para:
  1. Escrever o teste **primeiro**.
  2. Confirmar que ele **falha** antes da implementação.
  3. Só então implementar.
- Red flags:
  - Testes com assertivas fracas (`expect(x).toBeDefined()`).
  - Testes que mockam o próprio código sob teste.
  - Snapshots gerados sem revisão humana.
  - Teste de paridade que compara "com tolerância grande demais" (> `1e-4`) sem justificativa.

---

## 11. Loop de Correção

1. **Leia a mensagem de erro completa.** Stack trace inteira.
2. **Reproduza isoladamente.** Rode só o teste que falha.
3. **Formule hipótese.** Não chute; explique por que acha que falhou.
4. **Ajuste um detalhe.** Rode de novo.
5. Limite: **3 tentativas** sem progresso → escalar para humano.

**Proibido:** desabilitar teste, ajustar assertiva para "passar", adicionar `.skip`/`xit`/`pytest.skip` sem issue vinculada.

---

## 12. Checklist Rápido

Antes de marcar uma tarefa como pronta:

- [ ] Todo critério de aceitação do `PLAN.md` tem teste cobrindo.
- [ ] Caminho feliz, ao menos 2 bordas e uma falha esperada testados.
- [ ] Teste de paridade passa se o motor foi tocado.
- [ ] Testes rodam em < 30s localmente (suite completa unit+integr).
- [ ] Nenhum `.only`, `.skip`, `focus`, `fdescribe` deixado.
- [ ] Nenhum `print`/`console.log` nos testes.
- [ ] Dados sensíveis / PII não aparecem nem em fixture.
- [ ] CI verde em todos os checks.

---

## 13. Testes de Paridade Python ↔ TypeScript

> Categoria própria dado o motor duplicado. **Passar paridade é gate obrigatório** de qualquer PR que toca `engine/` ou `web/src/engine/`.

### 13.1 Como funciona

1. **Geração da fixture (Python).** `scripts/gen_fixture.py` roda uma partida com seed fixa e sequência de ações pré-definidas, grava cada step `(tick, state, action_left, action_right)` em `shared/fixtures/<nome>.json`.
2. **Verificação em Python.** `scripts/parity_check.py` re-executa a fixture no motor Python e compara com os estados gravados (sanity check; deve ser trivialmente igual).
3. **Verificação em TS.** `web/tests/parity.test.ts` carrega a mesma fixture, roda o motor TS com os mesmos inputs, e compara estado por estado.

### 13.2 Tolerâncias

- Valores inteiros (`paddle_*_y`, `score_*`, `tick`): igualdade estrita.
- Floats (`ball_*`): tolerância `1e-6`.
- Bitmaps: igualdade estrita (são `uint8` com valores em `{0, 1}`).

### 13.3 Fixtures obrigatórias

- `short_game_low_speed.json` — 300 steps, sem aceleração acumulada significativa.
- `long_game_accelerated.json` — 2000 steps, bola atinge ~80% de `max_ball_speed`.
- `max_speed_clamped.json` — 500 steps com bola iniciando próxima ao `max_ball_speed`, valida clamp.

Adicionar nova fixture sempre que um bug de paridade for encontrado — vira regressão permanente.

### 13.4 Regra de ouro

**Se a paridade quebrou, não é culpa do teste.** Um dos dois motores divergiu. Encontre onde, corrija em ambos, adicione fixture que cobre o caso.
