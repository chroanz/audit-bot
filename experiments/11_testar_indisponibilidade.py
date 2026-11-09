from pathlib import Path
from datetime import datetime, timezone
from urllib import request, error
from uuid import uuid4
import json

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

modelo = joblib.load(BASE / "models/modelo_seed42.joblib")

dados = pd.read_csv(BASE / "data/german_credit_auditoria.csv")

predicoes = pd.read_csv(RESULTS / "predicoes_seed42.csv")

colunas = list(modelo.feature_names_in_)

# Selecionar uma transação simulada
indice = int(predicoes.iloc[0]["indice_original"])
entrada = dados.iloc[[indice]][colunas]

# A decisão é produzida ANTES da tentativa de auditoria.
probabilidade = float(modelo.predict_proba(entrada)[0, 1])
classe = int(probabilidade >= 0.5)

# Comparar com a execução original.
original = predicoes.iloc[0]

assert np.isclose(
    probabilidade,
    original["probabilidade_bom_risco"],
    atol=1e-9,
    rtol=0,
)

assert classe == int(original["classe_prevista"])

print("=== CLASSIFICADOR ===")
print("Decisão produzida:", classe)
print("Probabilidade:", probabilidade)
print("Correspondência com experimento: OK")

# Criar uma nova tentativa de auditoria.
uuid = str(uuid4())

features = entrada.iloc[0].to_dict()

features = {
    k: v.item() if isinstance(v, np.generic) else v for k, v in features.items()
}

payload = {
    "transaction_id": uuid,
    "source_system": "german-credit-seed42",
    "features": features,
    "protected_attributes": {"faixa_etaria": str(original["grupo_idade"])},
    "prediction": classe,
    "score": probabilidade,
}

url = "http://127.0.0.1:8001/webhook/decisions"

print("\n=== TENTATIVA DE AUDITORIA ===")

falha_registrada = False

try:
    requisicao = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(requisicao, timeout=2):
        pass

    raise AssertionError(
        "A API respondeu. Interrompa o serviço da porta 8001 antes do teste."
    )

except (error.URLError, TimeoutError, ConnectionError) as exc:
    falha_registrada = True
    descricao_erro = str(exc)

    print("Serviço de auditoria indisponível.")
    print("Falha:", descricao_erro)

assert falha_registrada

# Registrar a falha somente no sistema cliente.
evidencia = {
    "horario_utc": datetime.now(timezone.utc).isoformat(),
    "transaction_id": uuid,
    "indice_original": indice,
    "prediction": classe,
    "score": probabilidade,
    "auditoria_recebida": False,
    "classificador_continuou": True,
    "erro_envio": descricao_erro,
}

arquivo = RESULTS / "teste_indisponibilidade.json"

arquivo.write_text(
    json.dumps(
        evidencia,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# Nova predição após a falha da auditoria.
probabilidade_depois = float(modelo.predict_proba(entrada)[0, 1])

assert np.isclose(
    probabilidade_depois,
    probabilidade,
    atol=1e-12,
    rtol=0,
)

print("\n=== RESULTADO ===")
print("Classificador continuou funcionando: SIM")
print("Predição preservada: SIM")
print("Falha de auditoria registrada: SIM")
print("Evidência:", arquivo)
print("\nTeste de indisponibilidade aprovado!")
