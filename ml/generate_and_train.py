"""
Gera um dataset sintético inspirado no German Credit Dataset e treina um
modelo simples (Regressao Logistica) que sera "auditado" pelo Audit-Bot.

Importante (propositalmente, para fins didaticos do MVP):
  - As variaveis protegidas (genero, regiao) NAO entram como feature do
    modelo -- exatamente como uma empresa real alegaria estar "sem viés".
  - Porem a renda_mensal (feature legitima) e gerada com uma distribuicao
    diferente por regiao/genero, funcionando como "proxy variable"
    (Art. 6, IX da LGPD / disparate impact). Isso recria em miniatura o
    problema descrito no documento: um algoritmo pode discriminar
    indiretamente mesmo sem usar o atributo sensivel diretamente.

O objetivo NAO e o modelo de credito (isso e so o "sistema de IA auditado"
de exemplo). O objetivo do projeto e o Audit-Bot, que roda ao lado dele.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATASET_PATH = BASE_DIR / "synthetic_credit_dataset.csv"

REGIOES = ["sudeste", "sul", "centro-oeste", "nordeste", "norte"]
GENEROS = ["M", "F"]

FEATURE_COLUMNS = [
    "idade",
    "renda_mensal",
    "valor_credito",
    "duracao_meses",
    "numero_dependentes",
    "score_bureau",
]
PROTECTED_COLUMNS = ["genero", "regiao"]


def _sample_income(regiao: str, genero: str, n: int) -> np.ndarray:
    """Renda mensal sintetica com disparidade regional/genero (proxy)."""
    regional_bias = {
        "sudeste": 1.00,
        "sul": 0.97,
        "centro-oeste": 0.93,
        "nordeste": 0.48,
        "norte": 0.45,
    }[regiao]
    gender_bias = 0.65 if genero == "F" else 1.00
    base = RNG.normal(4200, 1200, n)
    income = base * regional_bias * gender_bias
    return np.clip(income, 400, None)


def generate_dataset(n_samples: int = 6000) -> pd.DataFrame:
    regioes = RNG.choice(REGIOES, size=n_samples, p=[0.30, 0.20, 0.15, 0.22, 0.13])
    generos = RNG.choice(GENEROS, size=n_samples, p=[0.5, 0.5])

    idade = RNG.integers(18, 75, size=n_samples)
    duracao_meses = RNG.integers(6, 60, size=n_samples)
    numero_dependentes = RNG.integers(0, 5, size=n_samples)
    valor_credito = RNG.normal(8000, 3500, size=n_samples).clip(500, None)
    score_bureau = RNG.normal(620, 90, size=n_samples).clip(300, 850)

    renda_mensal = np.array(
        [
            _sample_income(r, g, 1)[0]
            for r, g in zip(regioes, generos)
        ]
    )

    df = pd.DataFrame(
        {
            "idade": idade,
            "renda_mensal": renda_mensal.round(2),
            "valor_credito": valor_credito.round(2),
            "duracao_meses": duracao_meses,
            "numero_dependentes": numero_dependentes,
            "score_bureau": score_bureau.round(1),
            "genero": generos,
            "regiao": regioes,
        }
    )

    # Regra "legitima" de aprovacao (comprometimento de renda + score),
    # sem uso direto de genero/regiao -- mas herdando o vies via renda.
    comprometimento = df["valor_credito"] / (df["duracao_meses"] * df["renda_mensal"] + 1)
    score_norm = (df["score_bureau"] - 300) / (850 - 300)

    logit = (
        2.4 * score_norm
        - 10.0 * comprometimento
        - 0.015 * df["numero_dependentes"]
        + 0.15
    )
    prob_aprovacao = 1 / (1 + np.exp(-logit))
    ruido = RNG.normal(0, 0.05, size=n_samples)
    df["aprovado"] = ((prob_aprovacao + ruido) > 0.5).astype(int)

    return df


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["aprovado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Acuracia (holdout) do modelo auditado: {acc:.3f}")
    return pipeline


def main():
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset sintetico salvo em {DATASET_PATH} ({len(df)} linhas)")

    model = train_model(df)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "protected_columns": PROTECTED_COLUMNS,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")

    # Relatorio rapido de disparidade "de fabrica" (antes de qualquer bot)
    df["predito"] = model.predict(df[FEATURE_COLUMNS])
    print("\nTaxa de aprovacao por regiao (evidencia do proxy embutido):")
    print(df.groupby("regiao")["predito"].mean().sort_values())
    print("\nTaxa de aprovacao por genero:")
    print(df.groupby("genero")["predito"].mean().sort_values())


if __name__ == "__main__":
    main()
