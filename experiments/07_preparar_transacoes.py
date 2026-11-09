from pathlib import Path
from uuid import uuid4
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
MODELS = BASE / "models"

modelo_path = MODELS / "modelo_seed42.joblib"
modelo = joblib.load(modelo_path)

dados = pd.read_csv(DATA / "german_credit_auditoria.csv")
predicoes = pd.read_csv(RESULTS / "predicoes_seed42.csv")

atributos = dados.drop(
    columns=[
        "Attribute13",
        "classe_original",
        "y",
        "grupo_idade",
    ]
)

# Confirmar que os resultados correspondem ao modelo
indices = predicoes["indice_original"].astype(int)
entradas = atributos.iloc[indices].copy()

probabilidades = modelo.predict_proba(entradas)[:, 1]
classes = (probabilidades >= 0.5).astype(int)

assert len(predicoes) == 300
assert np.allclose(
    probabilidades,
    predicoes["probabilidade_bom_risco"],
    atol=1e-12,
    rtol=0,
)
assert np.array_equal(classes, predicoes["classe_prevista"].to_numpy())

# Conferir os grupos etários
idades = dados.iloc[indices]["Attribute13"].to_numpy()

grupos = np.where(idades < 25, "Menos de 25", "25 ou mais")

assert np.array_equal(grupos, predicoes["grupo_idade"].to_numpy())

# Gerar mensagens no formato do webhook
payloads = []
vinculos = []

for posicao, (_, linha) in enumerate(predicoes.iterrows()):
    transaction_id = str(uuid4())

    features = {
        coluna: (valor.item() if isinstance(valor, np.generic) else valor)
        for coluna, valor in entradas.iloc[posicao].items()
    }

    payload = {
        "transaction_id": transaction_id,
        "source_system": "german-credit-seed42",
        "features": features,
        "protected_attributes": {"faixa_etaria": grupos[posicao]},
        "prediction": int(classes[posicao]),
        "score": float(probabilidades[posicao]),
    }

    assert "Attribute13" not in payload["features"]

    payloads.append(payload)

    vinculos.append(
        {
            "transaction_id": transaction_id,
            "indice_original": int(linha["indice_original"]),
            "classe_real": int(linha["classe_real"]),
            "classe_prevista": int(classes[posicao]),
            "grupo_idade": grupos[posicao],
        }
    )

# Salvar sem enviar à API
arquivo = RESULTS / "payloads_seed42.jsonl"

with arquivo.open("w", encoding="utf-8") as f:
    for payload in payloads:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

pd.DataFrame(vinculos).to_csv(RESULTS / "vinculos_seed42.csv", index=False)

hash_modelo = hashlib.sha256(modelo_path.read_bytes()).hexdigest()

manifesto = {
    "experimento": "German Credit",
    "semente": 42,
    "quantidade_transacoes": len(payloads),
    "modelo_sha256": hash_modelo,
    "arquivo_payloads": arquivo.name,
}

(RESULTS / "manifesto_seed42.json").write_text(
    json.dumps(manifesto, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n=== PREPARAÇÃO DA INTEGRAÇÃO ===")
print(f"Transações preparadas: {len(payloads)}")
print(f"UUIDs únicos: {len(set(p['transaction_id'] for p in payloads))}")
print(f"Modelo SHA-256: {hash_modelo[:16]}...")
print("\nDistribuição por grupo:")
print(pd.Series(grupos).value_counts())
print("\nArquivos preparados com sucesso.")
print("Nenhuma transação foi enviada à API.")
