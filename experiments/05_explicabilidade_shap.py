from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
MODELS = BASE / "models"

SEED = 42
TOLERANCIA = 1e-6

# 1. Recuperar modelo, dados e predições
modelo = joblib.load(MODELS / "modelo_seed42.joblib")
dados = pd.read_csv(DATA / "german_credit_auditoria.csv")
predicoes = pd.read_csv(RESULTS / "predicoes_seed42.csv")

X = dados.drop(columns=["Attribute13", "classe_original", "y", "grupo_idade"])
y = dados["y"].astype(int)

indices = np.arange(len(X))
idx_treino, idx_teste = train_test_split(
    indices, test_size=0.30, random_state=SEED, stratify=y
)

preprocessador = modelo.named_steps["preprocessador"]
classificador = modelo.named_steps["classificador"]

# 2. Selecionar até 100 registros de treinamento
rng = np.random.default_rng(SEED)
idx_referencia = rng.choice(idx_treino, size=min(100, len(idx_treino)), replace=False)

background = preprocessador.transform(X.iloc[idx_referencia])

if hasattr(background, "toarray"):
    background = background.toarray()

# 3. Criar explicador SHAP para regressão logística
explicador = shap.LinearExplainer(classificador, background)

# 4. Selecionar três casos distintos
favoraveis = predicoes[predicoes["classe_prevista"] == 1].sort_values(
    "probabilidade_bom_risco", ascending=False
)

desfavoraveis = predicoes[predicoes["classe_prevista"] == 0].sort_values(
    "probabilidade_bom_risco"
)

proximos = predicoes.assign(
    distancia=abs(predicoes["probabilidade_bom_risco"] - 0.5)
).sort_values("distancia")

selecionados = []
usados = set()

for tipo, candidatos in [
    ("Favorável", favoraveis),
    ("Desfavorável", desfavoraveis),
    ("Próxima ao limiar", proximos),
]:
    for _, linha in candidatos.iterrows():
        indice = int(linha["indice_original"])
        if indice not in usados:
            selecionados.append((tipo, linha))
            usados.add(indice)
            break
    else:
        raise ValueError(f"Não foi possível selecionar caso: {tipo}")

# 5. Mapear colunas transformadas aos atributos originais
encoder = preprocessador.named_transformers_["categoricas"]
categoricas = list(preprocessador.transformers_[0][2])
numericas = list(preprocessador.transformers_[1][2])

nomes_originais = []

for nome, categorias in zip(categoricas, encoder.categories_):
    nomes_originais.extend([nome] * len(categorias))

nomes_originais.extend(numericas)

assert len(nomes_originais) == len(preprocessador.get_feature_names_out()), (
    "Mapeamento dos atributos inconsistente."
)

# 6. Gerar e verificar explicações
linhas_contribuicoes = []
linhas_resumo = []

for tipo, caso in selecionados:
    indice = int(caso["indice_original"])

    entrada = X.iloc[[indice]]
    transformada = preprocessador.transform(entrada)

    if hasattr(transformada, "toarray"):
        transformada = transformada.toarray()

    explicacao = explicador(transformada)

    valores = np.asarray(explicacao.values).reshape(-1)

    referencia = float(np.asarray(explicacao.base_values).reshape(-1)[0])

    escore = float(classificador.decision_function(transformada)[0])

    reconstruido = referencia + float(valores.sum())
    erro = abs(escore - reconstruido)

    if erro > TOLERANCIA:
        raise AssertionError(f"Explicação inconsistente: erro={erro}")

    # Agregar contribuições das colunas one-hot
    contribuicoes = (
        pd.DataFrame({"atributo": nomes_originais, "shap_log_odds": valores})
        .groupby("atributo", as_index=False, sort=False)["shap_log_odds"]
        .sum()
    )

    contribuicoes["abs_shap"] = contribuicoes["shap_log_odds"].abs()

    contribuicoes = contribuicoes.sort_values("abs_shap", ascending=False)

    contribuicoes["caso"] = tipo
    contribuicoes["indice_original"] = indice

    linhas_contribuicoes.append(contribuicoes)

    linhas_resumo.append(
        {
            "caso": tipo,
            "indice_original": indice,
            "grupo_idade": caso["grupo_idade"],
            "classe_real": int(caso["classe_real"]),
            "classe_prevista": int(caso["classe_prevista"]),
            "probabilidade": float(caso["probabilidade_bom_risco"]),
            "valor_referencia_log_odds": referencia,
            "escore_log_odds": escore,
            "escore_reconstruido": reconstruido,
            "erro_reconstrucao": erro,
        }
    )

    print(f"\n=== CASO: {tipo} ===")
    print(f"Índice original: {indice}")
    print(f"Grupo: {caso['grupo_idade']}")
    print(f"Probabilidade: {caso['probabilidade_bom_risco']:.4f}")
    print(f"Escore log-odds: {escore:.6f}")
    print(f"Erro de reconstrução: {erro:.10f}")

    print("\nCinco maiores contribuições absolutas:")
    print(contribuicoes[["atributo", "shap_log_odds"]].head(5).to_string(index=False))

# 7. Salvar evidências
pd.concat(linhas_contribuicoes, ignore_index=True).to_csv(
    RESULTS / "shap_contribuicoes_seed42.csv", index=False
)

pd.DataFrame(linhas_resumo).to_csv(RESULTS / "shap_casos_seed42.csv", index=False)

print("\nExplicações SHAP salvas com sucesso.")
