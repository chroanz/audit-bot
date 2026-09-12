# Audit-Bot (MVP)

Middleware de **governança de IA e mitigação de viés algorítmico**, desenvolvido
como prova de conceito acadêmica com base no documento *"Governança de TI e
Mitigação de Viés em Inteligência Artificial"*.

O Audit-Bot audita, **de forma passiva e desacoplada**, as decisões de um
sistema de Machine Learning de terceiros: calcula métricas de equidade,
gera explicações individuais das decisões e emite alertas de conformidade
e relatórios exigidos pela LGPD e pelo CDC.

> Este é um MVP acadêmico. Não foi feita revisão de segurança para produção
> (autenticação, TLS, rate limiting etc. ficam fora do escopo do protótipo).

## Stack

Conforme sugerido no documento (RNF04 — interoperabilidade via REST/JSON):

- **Python 3.12 + FastAPI** — serviço único de auditoria (sem worker/fila).
- **SQLite** — log de auditoria *append-only*.
- **scikit-learn** — modelo simples (Regressão Logística) usado como o
  "sistema de IA auditado" de demonstração.
- **Aproximação linear de pesos** como motor de explicabilidade (RF04),
  com suporte opcional a **SHAP** se a biblioteca estiver instalada.
- **ReportLab** — geração do PDF de revisão (RF05 / Art. 20 LGPD).
- HTML/CSS/JS puro (sem dependências externas) para o painel de governança.

## Como o MVP mapeia os requisitos do documento

| Requisito | Onde está implementado |
|---|---|
| RF01 — Ingestão passiva via webhook | `POST /webhook/decisions` (`app/main.py`) |
| RF02 — Auditoria de disparidade | `app/fairness.py` (Disparate Impact Ratio, regra dos 4/5) |
| RF03 — Alertas de conformidade | `app/fairness.py` + tabela `alerts` (não bloqueia o usuário final, apenas notifica a governança — conforme o texto do RF03) |
| RF04 — Motor de explicabilidade | `app/explainability.py` ("Tradutor Cidadão") |
| RF05 — Relatório de revisão (Art. 20 LGPD) | `GET /transactions/{id}/report` + `app/reports.py` |
| RNF01 — Desacoplamento crítico | Serviço HTTP independente; se cair, não afeta o sistema de ML cliente (o cliente só faz um `POST` best-effort de auditoria) |
| RNF02 — Privacidade by design | `transaction_id` obrigatoriamente hash/UUID; payload rejeita campos como CPF, nome, e-mail (`app/schemas.py`, `app/config.py`) |
| RNF03 — Auditabilidade e imutabilidade | Tabelas `transactions`/`alerts`/`alert_reviews` em SQLite com *triggers* `BEFORE UPDATE/DELETE` que abortam a operação (`app/database.py`) |
| RNF04 — Interoperabilidade básica | API REST/JSON via FastAPI |

**Fora do escopo deste MVP** (deixado como trabalho futuro, e explicitamente
fora do RF03 do documento): o "Kill-Switch" de bloqueio em tempo real da
tabela de vulnerabilidades da Seção 4. Como a ingestão é passiva via
webhook (RF01/RNF01), o Audit-Bot só recebe uma *cópia* da decisão já
tomada — bloquear exigiria um proxy síncrono no caminho crítico, o que
contraria o desacoplamento exigido pelo RNF01.

## Estrutura

```
audit-bot/
  app/                 # o Audit-Bot em si (FastAPI)
    main.py            # rotas da API
    database.py         # SQLite append-only + triggers de imutabilidade
    fairness.py          # Disparate Impact / regra dos 4/5
    explainability.py    # explicações individuais (SHAP ou aproximação linear)
    reports.py            # geração do PDF (Art. 20 LGPD)
    schemas.py            # validação Pydantic (inclui bloqueio de PII)
    config.py              # limiares e constantes
  ml/
    generate_and_train.py # gera dataset sintético + treina o modelo auditado
  dashboard/
    index.html            # painel de governança (estático, sem build)
  scripts/
    simulate_traffic.py   # simula o sistema de ML cliente enviando decisões
  data/
    audit_log.db          # banco SQLite (criado em runtime)
  Dockerfile               # imagem da aplicacao (treina o modelo em build-time)
  docker-compose.yml       # sobe o Audit-Bot com volume persistente para o log
  entrypoint.sh            # aguarda o servidor subir e faz o seed automatico de demo
```

## Como rodar (Docker — recomendado para demonstrações)

Um único comando sobe o Audit-Bot **já com o painel de governança
populado**: o `entrypoint.sh` do container aguarda o servidor ficar
saudável e, se o log de auditoria estiver vazio, dispara automaticamente
300 decisões simuladas (o mesmo `scripts/simulate_traffic.py`), gerando
alertas reais de disparidade prontos para mostrar.

```bash
docker compose up -d --build
```

Acompanhe o log até aparecer `Audit-Bot pronto` (leva ~15-30s: build +
subida + seed automático):

```bash
docker compose logs -f audit-bot
```

Depois é só abrir:

- Painel de governança: **http://localhost:8000/dashboard/**
- Swagger (documentação interativa da API): **http://localhost:8000/docs**

Outros comandos úteis:

```bash
# Parar (mantém os dados no volume para a próxima demonstração)
docker compose down

# Resetar tudo (apaga o log de auditoria e força um novo seed automático
# na próxima subida) -- use entre uma demonstração e outra se quiser
# recomeçar do zero
docker compose down -v
docker compose up -d --build

# Gerar mais tráfego de demonstração manualmente a qualquer momento
# (o container já está rodando; isso só chama o script de dentro dele)
docker compose exec audit-bot python3 scripts/simulate_traffic.py --n 100 --url http://localhost:8000
```

O modelo "auditado" e o dataset sintético são gerados em **build-time**
(com seed fixa, então o resultado é sempre igual); só o log de auditoria
(`/app/data`, banco SQLite) é persistido em um volume Docker nomeado
(`audit_data`), sobrevivendo a `docker compose restart`/`up` mas sendo
removido por `docker compose down -v`.

## Como rodar sem Docker (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Gera o dataset sintético e treina o modelo de demonstração
#    (um "sistema de crédito" fictício, com viés de proxy embutido
#    de propósito: renda regional/gênero, sem usar o atributo protegido
#    diretamente no modelo -- assim como nos casos ViaQuatro/Decolar.com
#    citados no documento)
python ml/generate_and_train.py

# 2) Sobe o Audit-Bot
uvicorn app.main:app --reload --port 8000

# 3) Em outro terminal: simula o sistema cliente enviando decisões
python scripts/simulate_traffic.py --n 300 --url http://127.0.0.1:8000
```

Abra o painel de governança em **http://127.0.0.1:8000/dashboard/** — você
verá as taxas de aprovação por grupo protegido, os alertas de disparidade
disparados automaticamente e a lista de transações auditadas.

A documentação interativa da API (Swagger) fica em
**http://127.0.0.1:8000/docs**.

## Fluxo de ponta a ponta

1. O sistema de ML cliente (`scripts/simulate_traffic.py` simula esse
   papel) toma uma decisão de crédito e, **de forma assíncrona/best-effort**,
   envia uma cópia para `POST /webhook/decisions`: features não sensíveis,
   atributos protegidos (gênero, região), a predição e o score.
2. O Audit-Bot valida o payload (rejeita PII, exige `transaction_id`
   tipo hash/UUID), calcula os pesos de cada variável na decisão
   (RF04) e grava tudo de forma imutável (RNF03).
3. A cada nova transação com atributos protegidos, o Audit-Bot recalcula
   a taxa de decisões favoráveis por grupo e aplica a **regra dos 4/5**
   (Disparate Impact Ratio < 0.80). Se algum grupo estiver abaixo do
   limite, gera um alerta de governança (RF03) — sem bloquear o usuário
   final.
4. O DPO/Compliance acompanha os alertas pelo painel, registra uma
   revisão (nunca sobrescreve o alerta original — cria um novo registro
   em `alert_reviews`) e pode gerar, para qualquer transação, um PDF
   de revisão com a base legal do Art. 20 da LGPD.

## Testes manuais já validados

- Payload com `cpf`/`nome`/`email` → rejeitado com `422` (RNF02).
- `transaction_id` fora do formato hash/UUID → rejeitado com `422`.
- Reenvio do mesmo `transaction_id` → rejeitado com `409` (log é
  append-only, sem sobrescrita).
- `UPDATE`/`DELETE` diretos nas tabelas `transactions`/`alerts` →
  bloqueados no próprio SQLite pelos triggers de imutabilidade.
- Execução de `scripts/simulate_traffic.py` gera alertas reais de
  disparidade (o dataset sintético foi calibrado para reproduzir, em
  miniatura, o tipo de discriminação indireta via proxy discutida nos
  casos ViaQuatro e Decolar.com do documento).

## Próximos passos (fora do MVP)

- Autenticação/autorização entre o sistema cliente e o Audit-Bot.
- SHAP completo (já suportado via *fallback* automático — basta
  `pip install shap`) em vez da aproximação linear.
- Persistência mais robusta (Postgres) e fila para maior volume.
- Interceptação síncrona opcional (Kill-Switch) para quem aceitar
  acoplamento maior em troca de bloqueio em tempo real.
