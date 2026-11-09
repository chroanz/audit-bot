import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)

JOVEM = "Menos de 25"
REFERENCIA = "25 ou mais"
TOLERANCIA = 1e-6


def divisao(numerador, denominador):
    return numerador / denominador if denominador else np.nan


def calcular(y_true, y_pred, grupos):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    grupos = np.asarray(grupos)

    resultados = {}

    for nome in [JOVEM, REFERENCIA]:
        mascara = grupos == nome
        reais = y_true[mascara]
        previstos = y_pred[mascara]

        tp = int(np.sum((reais == 1) & (previstos == 1)))
        fn = int(np.sum((reais == 1) & (previstos == 0)))
        fp = int(np.sum((reais == 0) & (previstos == 1)))
        tn = int(np.sum((reais == 0) & (previstos == 0)))

        resultados[nome] = {
            "SR": divisao(tp + fp, len(reais)),
            "TPR": divisao(tp, tp + fn),
            "FPR": divisao(fp, fp + tn),
        }

    j = resultados[JOVEM]
    r = resultados[REFERENCIA]

    dpd = abs(j["SR"] - r["SR"])
    di = divisao(j["SR"], r["SR"])

    diferenca_tpr = abs(j["TPR"] - r["TPR"])
    diferenca_fpr = abs(j["FPR"] - r["FPR"])

    odds = (
        max(diferenca_tpr, diferenca_fpr)
        if np.isfinite([diferenca_tpr, diferenca_fpr]).all()
        else np.nan
    )

    return {
        "DPD": dpd,
        "DI": di,
        "igualdade_oportunidades": diferenca_tpr,
        "equalized_odds": odds,
        "alerta_DPD": bool(dpd > 0.10),
    }


def verificar(nome, reais, previstos, grupos, esperado):
    resultado = calcular(reais, previstos, grupos)

    print(f"\n=== {nome} ===")

    for metrica, valor_esperado in esperado.items():
        observado = resultado[metrica]

        if isinstance(valor_esperado, bool):
            assert observado is valor_esperado
        elif np.isnan(valor_esperado):
            assert np.isnan(observado)
        else:
            assert np.isclose(observado, valor_esperado, rtol=0, atol=TOLERANCIA)

        print(f"{metrica}: {observado} — OK")

    # Comparação independente quando todas
    # as taxas necessárias estão definidas.
    referencia = demographic_parity_difference(
        reais, previstos, sensitive_features=grupos
    )

    assert np.isclose(resultado["DPD"], referencia, rtol=0, atol=TOLERANCIA)

    if np.isfinite(resultado["igualdade_oportunidades"]):
        referencia = equal_opportunity_difference(
            reais, previstos, sensitive_features=grupos
        )
        assert np.isclose(
            resultado["igualdade_oportunidades"], referencia, rtol=0, atol=TOLERANCIA
        )

    if np.isfinite(resultado["equalized_odds"]):
        referencia = equalized_odds_difference(
            reais, previstos, sensitive_features=grupos
        )
        assert np.isclose(
            resultado["equalized_odds"], referencia, rtol=0, atol=TOLERANCIA
        )


grupos = [JOVEM] * 4 + [REFERENCIA] * 4

verificar(
    "Seleção igual",
    reais=[1, 0, 1, 0, 1, 0, 1, 0],
    previstos=[1, 0, 1, 0, 1, 0, 1, 0],
    grupos=grupos,
    esperado={
        "DPD": 0.0,
        "DI": 1.0,
        "igualdade_oportunidades": 0.0,
        "equalized_odds": 0.0,
        "alerta_DPD": False,
    },
)

verificar(
    "Disparidade conhecida",
    reais=[1, 0, 1, 0, 1, 0, 1, 0],
    previstos=[0, 0, 0, 0, 1, 1, 1, 1],
    grupos=grupos,
    esperado={
        "DPD": 1.0,
        "DI": 0.0,
        "igualdade_oportunidades": 1.0,
        "equalized_odds": 1.0,
        "alerta_DPD": True,
    },
)

verificar(
    "Ausência de positivos reais no grupo jovem",
    reais=[0, 0, 0, 0, 1, 0, 1, 0],
    previstos=[0, 0, 0, 0, 1, 0, 1, 0],
    grupos=grupos,
    esperado={
        "DPD": 0.5,
        "DI": 0.0,
        "igualdade_oportunidades": np.nan,
        "equalized_odds": np.nan,
        "alerta_DPD": True,
    },
)

verificar(
    "Taxa de referência igual a zero",
    reais=[1, 0, 1, 0, 1, 0, 1, 0],
    previstos=[1, 0, 0, 0, 0, 0, 0, 0],
    grupos=grupos,
    esperado={
        "DPD": 0.25,
        "DI": np.nan,
        "igualdade_oportunidades": 0.5,
        "equalized_odds": 0.5,
        "alerta_DPD": True,
    },
)

print("\nTODOS OS CASOS DE CONTROLE PASSARAM!")
