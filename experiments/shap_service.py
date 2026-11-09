from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/german_credit_auditoria.csv"

SEED = 42
TOLERANCIA = 1e-6


class GermanCreditExplainer:
    def __init__(self, modelo):
        self.modelo = modelo
        self.preprocessador = modelo.named_steps["preprocessador"]
        self.classificador = modelo.named_steps["classificador"]
        self.colunas = list(modelo.feature_names_in_)

        dados = pd.read_csv(DATA)
        y = dados["y"].astype(int)

        indices = np.arange(len(dados))

        idx_treino, _ = train_test_split(
            indices,
            test_size=0.30,
            random_state=SEED,
            stratify=y,
        )

        # Mesmo conjunto de referência usado
        # no experimento SHAP anterior.
        rng = np.random.default_rng(SEED)

        idx_referencia = rng.choice(
            idx_treino,
            size=min(100, len(idx_treino)),
            replace=False,
        )

        entradas = dados.iloc[idx_referencia][self.colunas]

        background = self.preprocessador.transform(entradas)

        if hasattr(background, "toarray"):
            background = background.toarray()

        self.explicador = shap.LinearExplainer(
            self.classificador,
            background,
        )

        encoder = self.preprocessador.named_transformers_["categoricas"]

        categoricas = list(self.preprocessador.transformers_[0][2])
        numericas = list(self.preprocessador.transformers_[1][2])

        nomes = []

        for atributo, categorias in zip(categoricas, encoder.categories_):
            nomes.extend([atributo] * len(categorias))

        nomes.extend(numericas)

        assert len(nomes) == len(self.preprocessador.get_feature_names_out())

        self.nomes_originais = nomes

    def explicar(self, features):
        entrada = pd.DataFrame(
            [features],
            columns=self.colunas,
        )

        transformada = self.preprocessador.transform(entrada)

        if hasattr(transformada, "toarray"):
            transformada = transformada.toarray()

        explicacao = self.explicador(transformada)

        contribuicoes = np.asarray(explicacao.values).reshape(-1)

        referencia = float(np.asarray(explicacao.base_values).reshape(-1)[0])

        escore = float(self.classificador.decision_function(transformada)[0])

        reconstruido = referencia + float(contribuicoes.sum())

        erro = abs(escore - reconstruido)

        if not np.isfinite(erro) or erro > TOLERANCIA:
            raise ValueError(f"Falha na reconstrução SHAP: {erro}")

        tabela = pd.DataFrame(
            {
                "atributo": self.nomes_originais,
                "contribuicao": contribuicoes,
            }
        )

        agregado = tabela.groupby("atributo", sort=False)["contribuicao"].sum()

        pesos = {nome: float(valor) for nome, valor in agregado.items()}

        return {
            "metodo": "SHAP LinearExplainer",
            "escala": "log-odds",
            "valor_referencia": referencia,
            "escore_linear": escore,
            "escore_reconstruido": reconstruido,
            "erro_reconstrucao": erro,
            "contribuicoes": pesos,
        }
