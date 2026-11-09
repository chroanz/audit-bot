from pathlib import Path

import numpy as np
import pandas as pd

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

dados = pd.read_csv(RESULTS / "predicoes_seed42.csv")
calculado = pd.read_csv(RESULTS / "equidade_metricas_seed42.csv").iloc[0]

y_true = dados["classe_real"]
y_pred = dados["classe_prevista"]
grupos = dados["grupo_idade"]

# Fairlearn utiliza, por padrão, a diferença
# entre a maior e a menor taxa de seleção.
metricas = {
    "DPD": demographic_parity_difference(y_true, y_pred, sensitive_features=grupos),
    "igualdade_oportunidades": equal_opportunity_difference(
        y_true, y_pred, sensitive_features=grupos
    ),
    "equalized_odds": equalized_odds_difference(
        y_true, y_pred, sensitive_features=grupos, agg="worst_case"
    ),
}

print("\n=== VALIDAÇÃO INDEPENDENTE ===")

nomes_colunas = {
    "DPD": "DPD",
    "igualdade_oportunidades": "diferenca_igualdade_oportunidades",
    "equalized_odds": "equalized_odds",
}

for nome, valor in metricas.items():
    coluna = nomes_colunas[nome]
    esperado = float(calculado[coluna])
    erro = abs(float(valor) - esperado)

    print(f"{nome}: Fairlearn={valor:.9f} | Manual={esperado:.9f} | Erro={erro:.12f}")

    assert np.isfinite(valor)
    assert erro <= 1e-6, f"Falha na métrica {nome}"

# O artigo define uma DI direcional:
# SR(menos de 25) / SR(25 ou mais).
# Essa razão não é necessariamente igual à
# razão simétrica usada pelo Fairlearn.

sr = dados.groupby("grupo_idade")["classe_prevista"].mean()

di_direcional = sr["Menos de 25"] / sr["25 ou mais"]

erro_di = abs(di_direcional - float(calculado["DI"]))

assert erro_di <= 1e-6

print(f"\nDI direcional: {di_direcional:.9f}")
print(f"Erro DI: {erro_di:.12f}")

print("\nTodas as métricas foram validadas!")
