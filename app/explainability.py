"""
RF04 (Motor de Explicabilidade): calcula o peso de cada variavel na
decisao automatizada -- o "Tradutor Cidadao" descrito no documento
(Art. 20 da LGPD), que transforma pesos matematicos em uma frase como
"Sua pontuacao caiu 40% devido a variavel X e 20% devido a Y".

Tenta usar SHAP quando a biblioteca estiver disponivel (fiel ao RF04 e
a referencia LUNDBERG; LEE, 2017). Se SHAP nao estiver instalado, cai
para uma aproximacao simples e transparente: contribuicao linear de
cada feature no espaco padronizado do modelo (coeficiente * z-score),
que e exatamente a alternativa que o proprio RF04 autoriza para o MVP.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependencia opcional
    _SHAP_AVAILABLE = False


class Explainer:
    def __init__(self, model_artifact: dict[str, Any]):
        self.pipeline = model_artifact["model"]
        self.feature_columns: list[str] = model_artifact["feature_columns"]
        self.scaler = self.pipeline.named_steps["scaler"]
        self.clf = self.pipeline.named_steps["clf"]
        self._shap_explainer = None
        if _SHAP_AVAILABLE:
            try:
                background = pd.DataFrame(
                    np.zeros((1, len(self.feature_columns))), columns=self.feature_columns
                )
                self._shap_explainer = shap.LinearExplainer(
                    self.clf, self.scaler.transform(background)
                )
            except Exception:
                self._shap_explainer = None

    @property
    def method(self) -> str:
        return "shap" if self._shap_explainer is not None else "linear_approximation"

    def explain(self, features: dict[str, float]) -> tuple[dict[str, float], str]:
        ordered = pd.DataFrame(
            [[features.get(col, 0.0) for col in self.feature_columns]],
            columns=self.feature_columns,
        )
        scaled = self.scaler.transform(ordered)

        if self._shap_explainer is not None:
            shap_values = self._shap_explainer.shap_values(scaled)[0]
            weights = {
                col: float(val) for col, val in zip(self.feature_columns, shap_values)
            }
        else:
            coefs = self.clf.coef_[0]
            contributions = coefs * scaled[0]
            weights = {
                col: float(val) for col, val in zip(self.feature_columns, contributions)
            }

        text = self._to_citizen_language(weights)
        return weights, text

    @staticmethod
    def _to_citizen_language(weights: dict[str, float]) -> str:
        total_abs = sum(abs(v) for v in weights.values()) or 1e-9
        ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)

        friendly_names = {
            "idade": "idade",
            "renda_mensal": "renda mensal informada",
            "valor_credito": "valor do credito solicitado",
            "duracao_meses": "prazo solicitado",
            "numero_dependentes": "numero de dependentes",
            "score_bureau": "pontuacao de credito (bureau)",
        }

        parts = []
        for name, weight in ranked[:3]:
            pct = abs(weight) / total_abs * 100
            direction = "contribuiu positivamente" if weight > 0 else "contribuiu negativamente"
            label = friendly_names.get(name, name)
            parts.append(f"{label} {direction} em {pct:.0f}% do peso da decisao")

        if not parts:
            return "Nao foi possivel calcular a contribuicao das variaveis."

        return (
            "Explicacao da decisao automatizada (Art. 20 da LGPD): "
            + "; ".join(parts)
            + "."
        )
