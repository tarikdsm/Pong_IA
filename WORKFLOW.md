# WORKFLOW.md

> Ciclo operacional que **todo agente** deve seguir neste projeto. Não é sugestão — é contrato.
>
> Fluxo canônico: **Explore → Specify → Critique → Plan → Implement → Verify → Commit → Review**.

---

## 1. Visão Geral do Ciclo

```
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Explore  │──▶│ Specify  │──▶│ Critique │──▶│   Plan   │
  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘
                                                    │
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────▼─────┐
  │ Review   │◀──│  Commit  │◀──│  Verify  │◀──│Implement │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                    │
                                   (loop TDD: teste─▶código─▶refator)
```

**Humano é porteiro em 3 pontos:** após Critique, após Plan, após Review.

---

## 2. Fase 1 — Explore

**Objetivo:** entender o território antes de mexer.

O agente:

- Lê `AGENTS.md`, `ARCHITECTURE.md`, `PLAN.md` (se existir) e `README.md`.
- Busca no repositório pelos módulos relacionados à tarefa.
- Identifica padrões e convenções já em uso.
- Se a tarefa toca `engine/`, abre `engine/pong_engine/` **e** `web/src/engine/` para conhecer os dois lados da paridade.

**Saída esperada:** resumo curto em prosa do contexto. **Sem código.**

**Regras:**

- Se a tarefa toca área desconhecida, Explore é obrigatório e mais longo.
- Não confie em memória de sessão anterior — releia.

---

## 3. Fase 2 — Specify

**Objetivo:** formular o que será feito em termos inequívocos.

O agente produz:

- Uma frase de intenção: *"Vou implementar X para resolver Y, tocando nos arquivos A, B e C."*
- Lista de requisitos funcionais e não-funcionais (seção 3 do `PLAN.md`).
- Critérios de aceitação testáveis (seção 4 do `PLAN.md`).

Se qualquer requisito estiver ambíguo, **pergunta antes de seguir**.

---

## 4. Fase 3 — Critique

> Passo frequentemente ignorado; é o que separa vibe coding de engenharia.

O agente **desafia o próprio pedido** e preenche a seção 2 do `PLAN.md`:

- A solução proposta faz sentido técnico?
- Já existe algo parecido no repositório?
- Existe alternativa mais simples/barata/segura?
- Quais riscos (segurança, performance, acoplamento, dados, **paridade Python↔TS**)?
- Quais suposições estão sendo feitas?

**Se encontrar problema sério, pare e leve ao humano.** Não siga "porque foi pedido".

---

## 5. Fase 4 — Plan

**Objetivo:** quebrar o trabalho em micro-passos pequenos, testáveis e independentes.

O agente:

- Preenche as seções 5 a 8 do `PLAN.md` (Design Técnico, Estratégia de Testes, Passos, Rollout).
- Cada passo de implementação deve ser **verificável isoladamente** (idealmente 1 commit, 1 teste).
- Lista arquivos que vão mudar e por quê.
- Se precisar de nova dependência, justifica.

**Regra específica deste projeto:** qualquer passo que tocar `engine/pong_engine/` **obriga** um passo equivalente em `web/src/engine/` no mesmo plano, mais um passo de verificação de paridade. Omiti-los é desvio do WORKFLOW.

**Gate humano:** o humano lê o `PLAN.md` e dá OK antes da implementação. Sem OK, não implementa.

---

## 6. Fase 5 — Implement (com TDD)

Para **cada passo** do plano, o agente executa o micro-loop:

```
  Red ─▶ Green ─▶ Refactor ─▶ próximo passo
```

1. **Red** — escreva o teste que falha pelo motivo certo. Rode e veja falhar.
2. **Green** — escreva o mínimo de código para passar. Rode e veja passar.
3. **Refactor** — limpe, extraia, renomeie. Rode de novo, tudo verde.

**Regras:**

- Um passo por vez. Não acumule 5 mudanças antes de rodar teste.
- **Diff cirúrgico:** toque só no necessário para o passo atual.
- Se descobrir que o plano estava errado, **volte a Plan**; não improvise no código.
- Não introduza dependência nova sem atualizar `PLAN.md` e revalidar com humano.
- **Ao mudar `engine/pong_engine/*.py`, escreva a mudança TS equivalente em `web/src/engine/*.ts` imediatamente a seguir.** Não acumule débito de paridade.

---

## 7. Fase 6 — Verify

Antes de declarar qualquer passo "pronto", o agente roda **todos** os checks:

```bash
uv run ruff check . && uv run pytest
pnpm --filter web typecheck && pnpm --filter web lint && pnpm --filter web test
python scripts/parity_check.py    # obrigatório se engine/ foi tocado
```

Se qualquer um falhar:

- **Loop de correção** (máximo 3 tentativas):
  1. Lê a saída do erro completa.
  2. Formula hipótese clara do motivo.
  3. Aplica a menor correção possível.
  4. Reexecuta.
- Se após 3 tentativas não resolveu, **para e escala**. Não chuta, não desabilita teste, não comenta código.

Se algo fora do escopo quebrou (ex.: teste não relacionado começa a falhar):

- **Pare.** Não "conserte de passagem". Reporte ao humano.

**Falha de paridade é crítica** e não se enquadra no loop de 3 tentativas — pare imediatamente, analise, reporte.

---

## 8. Fase 7 — Commit

Quando o passo está verde e limpo:

- **Conventional Commits:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `build`, `ci`.
- Um commit = **uma ideia**. Refactor separado de feature. Teste pode ir junto do código que testa.
- Mensagem no imperativo, curta no título (≤ 72 chars), detalhes no corpo se necessário.
- Scopes convencionados neste projeto: `engine`, `web`, `train`, `viz`, `parity`, `config`, `docs`.
- Exemplos:
  - `feat(engine): implementa aceleração da bola ao rebater em raquetes`
  - `test(parity): cobre clamp de max_ball_speed em long game`
  - `refactor(web/engine): extrai cálculo de colisão para função pura`
  - `docs(readme): atualiza quickstart com passo de gen_config`
- **Não force-push** em branches compartilhadas.
- **Nunca commite** segredos, `.env`, `node_modules`, `.venv`, `checkpoints/`, `*.pt`, `*.onnx` (gerados), etc.

---

## 9. Fase 8 — Review

Antes de abrir PR / pedir revisão humana:

- Roda `git diff` e **lê o próprio diff** linha por linha.
- Pergunta a si mesmo:
  - Alguma mudança está fora do escopo do `PLAN.md`?
  - Sobrou `console.log`, `print`, `TODO` sem issue?
  - Alguma duplicação apareceu que pode ser extraída?
  - Alguma função ficou grande demais?
  - Algum teste com assertiva fraca?
  - **Mudei `engine/` mas esqueci o port TS? O parity_check roda?**
- Atualiza documentação se mudou contrato público, configuração ou decisão arquitetural.
- Marca checklist final do `PLAN.md`.
- Abre PR com descrição que referencia o `PLAN.md`.

**Gate humano:** revisão de código humana é **obrigatória** para qualquer mudança em:

- Contrato do `GameState` (adicionar/remover campos).
- Formato de `shared/config.json`.
- Formato de exportação ONNX (shapes, nomes das saídas).
- Regras do motor que afetam fixtures existentes.
- Dependências novas.

---

## 10. Multi-Agent / Sub-Agentes

Se o projeto usa mais de um agente:

- **Separar papéis reduz débito.** Um único agente fazendo tudo tende a concordar consigo mesmo.
- Sugestão de papéis:
  - **Planner** — produz `PLAN.md`, não escreve código.
  - **Implementer** — segue o plano estritamente, não muda escopo.
  - **Reviewer** — lê o diff e busca pontos do `AGENTS.md` §13 que foram violados; confirma paridade.
  - **Tester** — foca em cobertura e qualidade dos testes (especialmente paridade).
- Cada papel recebe **apenas o contexto que precisa**.

---

## 11. Quando o Agente Deve Parar

Pare e escale ao humano quando:

- O `PLAN.md` precisaria ser alterado de forma não trivial.
- Um teste antes verde ficou vermelho fora do seu escopo.
- Teste de paridade falha por motivo não óbvio.
- A tarefa exige decisão de modelagem ML (arquitetura, hiperparâmetros importantes).
- A mudança afeta formato ONNX ou contrato compartilhado.
- Loop de correção atingiu 3 tentativas sem progresso.
- Encontrou código existente que parece ter bug — **não conserte silenciosamente**; reporte.
- O humano disse "segue" mas as condições mudaram.

**Parar é um sinal de qualidade, não de falha.**

---

## 12. Anti-Padrões a Evitar

- ❌ "Deixa eu só ajustar isso aqui também" (scope creep).
- ❌ "Vou desabilitar esse teste chato e volto depois" (nunca volta).
- ❌ "Ajustei a tolerância pra 1e-2 e a paridade passou" (mascaramento de bug).
- ❌ "O tipo tá difícil, `any` resolve" (débito técnico imediato).
- ❌ "Esse try/catch silencia erro mas passa o teste" (bomba-relógio).
- ❌ "Vou reescrever esse módulo porque acho feio" (fora do escopo).
- ❌ "Mudei só o Python, a gente portra TS depois" (débito de paridade).
- ❌ "O agente anterior disse que estava pronto" (verifique).
- ❌ "Funcionou uma vez, devo commitar antes que quebre" (flaky é bug).

---

## 13. Cheatsheet para Colar no Prompt

Quando iniciar uma sessão com o agente, cole algo assim:

```
Siga estritamente o WORKFLOW.md deste repositório.

Tarefa: <descrição>.

Execute na ordem: Explore → Specify → Critique → Plan.
PARE após o Plan e mostre o PLAN.md atualizado para minha aprovação.
Não escreva código de produção antes do meu OK.

Regras-chave:
- TDD obrigatório: teste vermelho antes de qualquer código.
- Diff cirúrgico, mínimo necessário.
- Paridade Python↔TS: qualquer mudança em engine/ exige mudança equivalente
  em web/src/engine/ e parity_check verde.
- Checks (ruff + pytest + typecheck + eslint + vitest + parity) verdes antes de commit.
- Máximo 3 tentativas no loop de correção; depois, escalar.
- Nunca desabilitar teste, afrouxar tolerância de paridade, nem silenciar erro.
```
