# CLAUDE.md

Este projeto segue o padrão aberto **AGENTS.md**. Todas as regras, stack, comandos e convenções estão em [AGENTS.md](AGENTS.md) — leia-o primeiro.

**Ordem de leitura recomendada para um novo agente:**

1. [AGENTS.md](AGENTS.md) — identidade, stack, comandos, convenções.
2. [WORKFLOW.md](WORKFLOW.md) — ciclo de trabalho obrigatório.
3. [ARCHITECTURE.md](ARCHITECTURE.md) — decisões arquiteturais e ADRs.
4. [PLAN.md](PLAN.md) — plano da tarefa atual.
5. [TESTING.md](TESTING.md) — contrato de testes (em especial §13, paridade Python↔TS).

**Regra crítica deste projeto:** o motor de Pong existe em Python (canônico) e TypeScript (browser) e a paridade entre os dois é gate de commit. Veja ADR-0001 em [ARCHITECTURE.md](ARCHITECTURE.md) e §7 de [AGENTS.md](AGENTS.md).
