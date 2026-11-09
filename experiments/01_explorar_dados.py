from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Diretórios do experimento
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Carregamento da versão original do German Credit Data
dataset = fetch_ucirepo(id=144)

X = dataset.data.features.copy()
y_original = dataset.data.targets.iloc[:, 0].astype(int)

# Verificação da estrutura
assert X.shape == (1000, 20), f"Dimensões inesperadas: {X.shape}"

assert set(y_original.unique()) == {1, 2}, "Rótulos diferentes dos esperados."

# Attribute13 corresponde à idade, em anos
assert "Attribute13" in X.columns
idade = pd.to_numeric(X["Attribute13"], errors="raise")

# Conforme a metodologia do artigo:
# original 1 = bom risco -> y = 1
# original 2 = mau risco -> y = 0
y = y_original.map({1: 1, 2: 0})

grupo = idade.map(lambda valor: "Menos de 25" if valor < 25 else "25 ou mais")

# Base de auditoria: mantém idade e grupo
dados = X.copy()
dados["classe_original"] = y_original.to_numpy()
dados["y"] = y.to_numpy()
dados["grupo_idade"] = grupo.to_numpy()

# Base do classificador: sem idade ou grupo
X_modelo = X.drop(columns=["Attribute13"])

# Resumo dos grupos
resumo = pd.crosstab(dados["grupo_idade"], dados["y"]).reindex(
    index=["Menos de 25", "25 ou mais"], columns=[0, 1], fill_value=0
)

resumo.columns = ["Mau risco (0)", "Bom risco (1)"]
resumo["Total"] = resumo.sum(axis=1)
resumo["Taxa de bom risco"] = resumo["Bom risco (1)"] / resumo["Total"]

print("\n=== GERMAN CREDIT DATA ===")
print(f"Registros: {len(X)}")
print(f"Atributos originais: {X.shape[1]}")
print(f"Atributos usados pelo modelo: {X_modelo.shape[1]}")
print(f"Valores ausentes: {X.isna().sum().sum()}")

print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")
print(y.value_counts().sort_index().to_string())

print("\n=== DISTRIBUIÇÃO POR IDADE ===")
print(grupo.value_counts().to_string())

print("\n=== CLASSES POR GRUPO ===")
print(resumo.to_string())

print("\n=== TIPOS DOS ATRIBUTOS ===")
print(X.dtypes.to_string())

# Exportação dos dados e evidências
dados.to_csv(DATA_DIR / "german_credit_auditoria.csv", index=False)
X_modelo.to_csv(DATA_DIR / "atributos_modelo.csv", index=False)
resumo.to_csv(RESULTS_DIR / "caracterizacao_grupos.csv")

print("\nArquivos CSV gerados com sucesso.")
