from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

ARQUIVO = RESULTS_DIR / "predicoes_seed42.csv"

JOVENS = "Menos de 25"
REFERENCIA = "25 ou mais"

LIMIAR_ALERTA = 0.10

# ============================================
# 1. CARREGAR AS PREDIÇÕES
# ============================================

dados = pd.read_csv(ARQUIVO)

assert len(dados) == 300
assert set(dados["grupo_idade"]) == {JOVENS, REFERENCIA}

# ============================================
# 2. CALCULAR MÉTRICAS POR GRUPO
# ============================================


def divisao_segura(numerador, denominador):
    if denominador == 0:
        return float("nan")
    return numerador / denominador


resultados = []

for grupo in [JOVENS, REFERENCIA]:
    df = dados[dados["grupo_idade"] == grupo]

    tn, fp, fn, tp = confusion_matrix(
        df["classe_real"], df["classe_prevista"], labels=[0, 1]
    ).ravel()

    total = len(df)

    taxa_selecao = divisao_segura(tp + fp, total)

    tpr = divisao_segura(tp, tp + fn)

    fpr = divisao_segura(fp, fp + tn)

    resultados.append(
        {
            "grupo": grupo,
            "total": total,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "positivos_reais": int(tp + fn),
            "negativos_reais": int(fp + tn),
            "selecionados": int(tp + fp),
            "taxa_selecao": taxa_selecao,
            "TPR": tpr,
            "FPR": fpr,
            "amostra_reduzida": total < 30,
        }
    )

tabela = pd.DataFrame(resultados)
tabela = tabela.set_index("grupo")

# ============================================
# 3. MÉTRICAS DE EQUIDADE
# ============================================

jovens = tabela.loc[JOVENS]
referencia = tabela.loc[REFERENCIA]

sr_jovens = jovens["taxa_selecao"]
sr_referencia = referencia["taxa_selecao"]

DPD = abs(sr_jovens - sr_referencia)

DI = divisao_segura(sr_jovens, sr_referencia)

diferenca_TPR = abs(jovens["TPR"] - referencia["TPR"])

diferenca_FPR = abs(jovens["FPR"] - referencia["FPR"])

equalized_odds = (
    max(diferenca_TPR, diferenca_FPR)
    if np.isfinite([diferenca_TPR, diferenca_FPR]).all()
    else float("nan")
)

alerta = bool(np.isfinite(DPD) and DPD > LIMIAR_ALERTA)

metricas = {
    "semente": 42,
    "DPD": float(DPD),
    "DI": float(DI),
    "diferenca_igualdade_oportunidades": float(diferenca_TPR),
    "diferenca_FPR": float(diferenca_FPR),
    "equalized_odds": float(equalized_odds),
    "limiar_alerta": LIMIAR_ALERTA,
    "alerta_DPD": alerta,
}

# ============================================
# 4. EXIBIR RESULTADOS
# ============================================

print("\n=== MÉTRICAS POR GRUPO ===")
print(tabela.to_string())

print("\n=== INDICADORES DE EQUIDADE ===")

for nome, valor in metricas.items():
    print(f"{nome}: {valor}")

print("\n=== INTERPRETAÇÃO DO ALERTA ===")

if alerta:
    print("ALERTA: DPD superior ao limite operacional de 0.10.")
else:
    print("DPD não ultrapassou o limite operacional de 0.10.")

if any(tabela["amostra_reduzida"]):
    print("ATENÇÃO: existe grupo com menos de 30 registros.")

# ============================================
# 5. SALVAR EVIDÊNCIAS
# ============================================

tabela.to_csv(RESULTS_DIR / "equidade_grupos_seed42.csv")

pd.DataFrame([metricas]).to_csv(
    RESULTS_DIR / "equidade_metricas_seed42.csv", index=False
)

with open(RESULTS_DIR / "equidade_seed42.json", "w", encoding="utf-8") as arquivo:
    json.dump(metricas, arquivo, indent=2, ensure_ascii=False, allow_nan=False)

print("\nArquivos de equidade gerados com sucesso.")
