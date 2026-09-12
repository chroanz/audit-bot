"""
RF02 (Auditoria de Disparidade) e RF03 (Geracao de Alertas de
Conformidade).

Metrica usada: Disparate Impact Ratio, a mesma citada na tabela do
documento e na literatura de fairness (BAROCAS; SELBST, 2016):

    DI(grupo) = taxa_favoravel(grupo) / taxa_favoravel(grupo_referencia)

O grupo de referencia e o grupo com a MAIOR taxa de decisoes favoraveis
observada para aquele atributo protegido (abordagem "regra dos 4/5" /
"four-fifths rule", DI < 0.8 indica impacto desproporcional).

Conforme RF03, o MVP NAO bloqueia a decisao do usuario final (isso exigiria
interceptar o fluxo em tempo real, o que contraria o RNF01 de
desacoplamento critico via webhook passivo). Em vez disso, dispara um
alerta no painel de governanca para a equipe de engenharia revisar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app import database as db
from app.config import DISPARATE_IMPACT_THRESHOLD, MIN_GROUP_SAMPLE_SIZE


@dataclass
class FairnessBreach:
    protected_attribute: str
    group_value: str
    reference_value: str
    disparate_impact_ratio: float
    group_sample_size: int
    reference_favorable_rate: float
    group_favorable_rate: float


def evaluate_protected_attribute(protected_attribute: str) -> list[FairnessBreach]:
    stats = db.get_group_stats(protected_attribute)
    if len(stats) < 2:
        return []

    parsed = [
        {
            "group_value": row["group_value"],
            "n": row["n"],
            "favorable_rate": row["favorable_rate"] or 0.0,
        }
        for row in stats
        if row["group_value"] is not None
    ]
    if len(parsed) < 2:
        return []

    reference = max(parsed, key=lambda g: g["favorable_rate"])
    breaches: list[FairnessBreach] = []

    for group in parsed:
        if group["group_value"] == reference["group_value"]:
            continue
        if group["n"] < MIN_GROUP_SAMPLE_SIZE:
            continue
        if reference["favorable_rate"] <= 0:
            continue

        ratio = group["favorable_rate"] / reference["favorable_rate"]
        if ratio < DISPARATE_IMPACT_THRESHOLD:
            breaches.append(
                FairnessBreach(
                    protected_attribute=protected_attribute,
                    group_value=str(group["group_value"]),
                    reference_value=str(reference["group_value"]),
                    disparate_impact_ratio=round(ratio, 4),
                    group_sample_size=group["n"],
                    reference_favorable_rate=round(reference["favorable_rate"], 4),
                    group_favorable_rate=round(group["favorable_rate"], 4),
                )
            )
    return breaches


def run_fairness_checks(latest_transaction_id: str) -> list[dict]:
    """Roda a checagem de disparidade para todos os atributos protegidos
    observados ate agora e registra novos alertas quando necessario.

    Retorna a lista de alertas efetivamente criados nesta chamada (para
    incluir na resposta HTTP do webhook, dando visibilidade imediata ao
    sistema cliente / operador).
    """
    created_alerts: list[dict] = []

    for attribute in db.list_distinct_protected_attributes():
        for breach in evaluate_protected_attribute(attribute):
            if db.has_recent_open_alert(attribute, breach.group_value):
                continue

            message = (
                f"Possivel viés discriminatório detectado (LGPD Art. 6º, IX): "
                f"o grupo '{breach.group_value}' do atributo '{attribute}' tem taxa de "
                f"decisões favoráveis de {breach.group_favorable_rate:.0%}, contra "
                f"{breach.reference_favorable_rate:.0%} do grupo de referência "
                f"'{breach.reference_value}' (razão de impacto disparate = "
                f"{breach.disparate_impact_ratio:.2f}, abaixo do limite de "
                f"{DISPARATE_IMPACT_THRESHOLD:.2f}). Recomenda-se pausar/revisar o "
                f"modelo e notificar o DPO."
            )

            alert_id = db.insert_alert(
                alert_type="disparate_impact",
                protected_attribute=attribute,
                group_value=breach.group_value,
                reference_value=breach.reference_value,
                disparate_impact_ratio=breach.disparate_impact_ratio,
                threshold=DISPARATE_IMPACT_THRESHOLD,
                group_sample_size=breach.group_sample_size,
                message=message,
                sample_transaction_ids=[latest_transaction_id],
            )
            created_alerts.append(
                {
                    "alert_id": alert_id,
                    "protected_attribute": attribute,
                    "group_value": breach.group_value,
                    "disparate_impact_ratio": breach.disparate_impact_ratio,
                    "message": message,
                }
            )

    return created_alerts
