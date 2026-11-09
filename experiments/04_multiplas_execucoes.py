from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "results"
RESULTADOS.mkdir(exist_ok=True)

dados = pd.read_csv(BASE / "data/german_credit_auditoria.csv")

X = dados.drop(columns=["Attribute13", "classe_original", "y", "grupo_idade"])
y = dados["y"].astype(int)
grupos = dados["grupo_idade"]

categoricas = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()

numericas = X.select_dtypes(include=["number"]).columns.tolist()


def taxa(numerador, denominador):
    return numerador / denominador if denominador > 0 else np.nan


def metricas_grupo(reais, previstas):
    tn, fp, fn, tp = confusion_matrix(reais, previstas, labels=[0, 1]).ravel()

    return {
        "n": len(reais),
        "positivos_reais": int(tp + fn),
        "negativos_reais": int(fp + tn),
        "SR": taxa(tp + fp, len(reais)),
        "TPR": taxa(tp, tp + fn),
        "FPR": taxa(fp, fp + tn),
    }


execucoes = []
grupos_execucoes = []

for seed in range(42, 52):
    indices = np.arange(len(X))

    treino, teste = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    preprocessador = ColumnTransformer(
        [
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                categoricas,
            ),
            (
                "numericas",
                StandardScaler(),
                numericas,
            ),
        ]
    )

    modelo = Pipeline(
        [
            ("preprocessador", preprocessador),
            (
                "classificador",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo.fit(X.iloc[treino], y.iloc[treino])

    convergencia = any(
        issubclass(aviso.category, ConvergenceWarning) for aviso in avisos
    )

    probabilidades = modelo.predict_proba(X.iloc[teste])[:, 1]

    previstas = (probabilidades >= 0.5).astype(int)
    reais = y.iloc[teste].to_numpy()
    grupos_teste = grupos.iloc[teste].to_numpy()

    por_grupo = {}

    for nome in ["Menos de 25", "25 ou mais"]:
        mascara = grupos_teste == nome

        resultado = metricas_grupo(
            reais[mascara],
            previstas[mascara],
        )

        por_grupo[nome] = resultado

        grupos_execucoes.append(
            {
                "semente": seed,
                "grupo": nome,
                **resultado,
            }
        )

    jovens = por_grupo["Menos de 25"]
    adultos = por_grupo["25 ou mais"]

    dpd = abs(jovens["SR"] - adultos["SR"])
    di = taxa(jovens["SR"], adultos["SR"])
    diff_tpr = abs(jovens["TPR"] - adultos["TPR"])
    diff_fpr = abs(jovens["FPR"] - adultos["FPR"])

    odds = (
        max(diff_tpr, diff_fpr) if np.isfinite([diff_tpr, diff_fpr]).all() else np.nan
    )

    execucoes.append(
        {
            "semente": seed,
            "acuracia": accuracy_score(reais, previstas),
            "precisao": precision_score(reais, previstas, zero_division=0),
            "recall": recall_score(reais, previstas, zero_division=0),
            "f1": f1_score(reais, previstas, zero_division=0),
            "DPD": dpd,
            "DI": di,
            "igualdade_oportunidades": diff_tpr,
            "diferenca_FPR": diff_fpr,
            "equalized_odds": odds,
            "alerta_DPD": bool(dpd > 0.10),
            "aviso_convergencia": convergencia,
        }
    )

    print(
        f"Semente {seed} | "
        f"Acurácia: {execucoes[-1]['acuracia']:.4f} | "
        f"DPD: {dpd:.4f} | "
        f"Alerta: {dpd > 0.10}"
    )

tabela = pd.DataFrame(execucoes)
tabela_grupos = pd.DataFrame(grupos_execucoes)

colunas = [
    "acuracia",
    "precisao",
    "recall",
    "f1",
    "DPD",
    "DI",
    "igualdade_oportunidades",
    "diferenca_FPR",
    "equalized_odds",
]

resumo = tabela[colunas].agg(["mean", "std"]).T
resumo.columns = ["media", "desvio_padrao"]

tabela.to_csv(RESULTADOS / "dez_execucoes.csv", index=False)
tabela_grupos.to_csv(RESULTADOS / "grupos_dez_execucoes.csv", index=False)
resumo.to_csv(RESULTADOS / "resumo_dez_execucoes.csv")

print("\n=== RESULTADOS DAS DEZ EXECUÇÕES ===")
print(tabela.to_string(index=False))

print("\n=== MÉDIA E DESVIO PADRÃO ===")
print(resumo.to_string())

print("\n=== ALERTAS ===")
print(f"Execuções com alerta: {tabela['alerta_DPD'].sum()}/10")
print(f"Execuções com aviso de convergência: {tabela['aviso_convergencia'].sum()}/10")

print("\n=== TAMANHO DO GRUPO MAIS JOVEM ===")
jovens = tabela_grupos[tabela_grupos["grupo"] == "Menos de 25"]
print(
    jovens[["semente", "n", "positivos_reais", "negativos_reais"]].to_string(
        index=False
    )
)

print("\nArquivos gerados com sucesso.")
