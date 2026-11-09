from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================
# 1. CONFIGURAÇÕES DO EXPERIMENTO
# ============================================

SEED = 42
TEST_SIZE = 0.30
THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. CARREGAMENTO DOS DADOS
# ============================================

dados = pd.read_csv(DATA_DIR / "german_credit_auditoria.csv")

# Idade fica disponível para auditoria,
# mas não é utilizada no treinamento.
X = dados.drop(
    columns=[
        "Attribute13",
        "classe_original",
        "y",
        "grupo_idade",
    ]
)

y = dados["y"].astype(int)
grupos = dados["grupo_idade"]

# ============================================
# 3. DIVISÃO TREINO / TESTE
# ============================================

indices = np.arange(len(dados))

idx_train, idx_test = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

X_train = X.iloc[idx_train]
X_test = X.iloc[idx_test]

y_train = y.iloc[idx_train]
y_test = y.iloc[idx_test]

grupos_test = grupos.iloc[idx_test]

# ============================================
# 4. PRÉ-PROCESSAMENTO
# ============================================

categoricas = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()

numericas = X.select_dtypes(include=["number"]).columns.tolist()

preprocessador = ColumnTransformer(
    transformers=[
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

# ============================================
# 5. REGRESSÃO LOGÍSTICA
# ============================================

modelo = Pipeline(
    steps=[
        ("preprocessador", preprocessador),
        (
            "classificador",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=SEED,
            ),
        ),
    ]
)

modelo.fit(X_train, y_train)

# ============================================
# 6. PREDIÇÕES
# ============================================

probabilidades = modelo.predict_proba(X_test)[:, 1]

predicoes = (probabilidades >= THRESHOLD).astype(int)

# ============================================
# 7. MÉTRICAS DE DESEMPENHO
# ============================================

metricas = {
    "semente": SEED,
    "registros_treino": len(X_train),
    "registros_teste": len(X_test),
    "acuracia": accuracy_score(y_test, predicoes),
    "precisao": precision_score(y_test, predicoes, zero_division=0),
    "recall": recall_score(y_test, predicoes, zero_division=0),
    "f1": f1_score(y_test, predicoes, zero_division=0),
}

matriz = confusion_matrix(
    y_test,
    predicoes,
    labels=[0, 1],
)

print("\n=== RESULTADOS DO MODELO ===")

for nome, valor in metricas.items():
    print(f"{nome}: {valor}")

print("\n=== MATRIZ DE CONFUSÃO ===")
print(matriz)

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(
    classification_report(
        y_test,
        predicoes,
        labels=[0, 1],
        target_names=["Mau risco", "Bom risco"],
        zero_division=0,
    )
)

# ============================================
# 8. EXPORTAÇÃO DAS EVIDÊNCIAS
# ============================================

resultados = pd.DataFrame(
    {
        "indice_original": idx_test,
        "idade": dados.iloc[idx_test]["Attribute13"].to_numpy(),
        "grupo_idade": grupos_test.to_numpy(),
        "classe_real": y_test.to_numpy(),
        "probabilidade_bom_risco": probabilidades,
        "classe_prevista": predicoes,
    }
)

resultados.to_csv(
    RESULTS_DIR / "predicoes_seed42.csv",
    index=False,
)

pd.DataFrame([metricas]).to_csv(
    RESULTS_DIR / "desempenho_seed42.csv",
    index=False,
)

pd.DataFrame(
    matriz,
    index=["Real 0", "Real 1"],
    columns=["Previsto 0", "Previsto 1"],
).to_csv(RESULTS_DIR / "matriz_confusao_seed42.csv")

with open(
    RESULTS_DIR / "configuracao_seed42.json",
    "w",
    encoding="utf-8",
) as arquivo:
    json.dump(
        {
            "semente": SEED,
            "tamanho_teste": TEST_SIZE,
            "limiar": THRESHOLD,
            "variaveis_categoricas": categoricas,
            "variaveis_numericas": numericas,
            "idade_excluida": True,
        },
        arquivo,
        indent=2,
        ensure_ascii=False,
    )

joblib.dump(
    modelo,
    MODELS_DIR / "modelo_seed42.joblib",
)

print("\n=== ARQUIVOS GERADOS ===")
print("Predições, métricas, matriz de confusão,")
print("configurações e modelo treinado salvos.")

print("\n=== DISTRIBUIÇÃO DOS GRUPOS NO TESTE ===")
print(resultados["grupo_idade"].value_counts())
