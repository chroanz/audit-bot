from pathlib import Path
from urllib import request, error
from collections import Counter
import json

BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "results/payloads_seed42.jsonl"

URL = "http://127.0.0.1:8001/webhook/decisions"
HEALTH = "http://127.0.0.1:8001/health"
SUMMARY = "http://127.0.0.1:8001/governance/summary"


def consultar(url):
    with request.urlopen(url, timeout=15) as resposta:
        return json.load(resposta)


payloads = [
    json.loads(linha)
    for linha in ARQUIVO.read_text(encoding="utf-8").splitlines()
    if linha.strip()
]

assert len(payloads) == 300
assert len({p["transaction_id"] for p in payloads}) == 300

estado_inicial = consultar(HEALTH)

if estado_inicial["transactions_audited"] not in (0, 1):
    raise RuntimeError(
        "Banco experimental já contém mais de uma transação. "
        "Não é seguro presumir que todos os registros são deste lote."
    )

contagem = Counter()
falhas = []

for numero, payload in enumerate(payloads, start=1):
    requisicao = request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(requisicao, timeout=30) as resposta:
            resultado = json.load(resposta)

        if (
            resultado.get("stored") is True
            and resultado.get("transaction_id") == payload["transaction_id"]
        ):
            contagem["inseridas"] += 1
        else:
            contagem["resposta_inesperada"] += 1
            falhas.append((numero, resultado))

    except error.HTTPError as excecao:
        corpo = excecao.read().decode("utf-8")

        if excecao.code == 409:
            contagem["duplicatas"] += 1
        else:
            contagem["erros_http"] += 1
            falhas.append((numero, excecao.code, corpo))

    except Exception as excecao:
        contagem["erros"] += 1
        falhas.append((numero, str(excecao)))

    if numero % 50 == 0:
        print(f"Processadas {numero}/300 transações")

estado_final = consultar(HEALTH)
resumo = consultar(SUMMARY)

print("\n=== RESULTADO DO ENVIO ===")
print(f"Inseridas: {contagem['inseridas']}")
print(f"Duplicatas: {contagem['duplicatas']}")
print(f"Erros HTTP: {contagem['erros_http']}")
print(f"Outros erros: {contagem['erros']}")
print(f"Respostas inesperadas: {contagem['resposta_inesperada']}")
print(f"Registros no banco: {estado_final['transactions_audited']}")

print("\n=== RESUMO DO AUDIT-BOT ===")
print(json.dumps(resumo, indent=2, ensure_ascii=False))

if falhas:
    print("\nPrimeiras falhas:")
    for falha in falhas[:5]:
        print(falha)

if estado_final["transactions_audited"] != 300:
    raise AssertionError("Quantidade final diferente de 300")

if any(contagem[k] for k in ("erros_http", "erros", "resposta_inesperada")):
    raise AssertionError("Ocorreram erros no envio")

print("\nLote enviado e contagem verificada!")
