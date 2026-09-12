"""
Audit-Bot -- MVP

Middleware de governanca de IA que audita, de forma PASSIVA e
DESACOPLADA (RNF01), as decisoes de um sistema de Machine Learning,
calculando metricas de equidade (RF02), gerando alertas de conformidade
(RF03), explicando decisoes individuais (RF04) e emitindo relatorios
para atender ao Art. 20 da LGPD (RF05).

Como rodar:
    uvicorn app.main:app --reload --port 8000

Endpoints principais:
    POST /webhook/decisions            -> RF01 + RF02 + RF04 + RF03
    GET  /governance/alerts            -> painel de governanca (RF03)
    POST /governance/alerts/{id}/review-> DPO registra revisao do alerta
    GET  /governance/transactions      -> lista transacoes recentes
    GET  /governance/summary           -> metricas agregadas de equidade
    GET  /transactions/{id}            -> detalhe de uma transacao
    GET  /transactions/{id}/report     -> PDF (RF05, Art. 20 LGPD)
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import database as db
from app import fairness
from app.config import BASE_DIR, MODEL_PATH
from app.explainability import Explainer
from app.reports import build_transaction_report
from app.schemas import AlertReviewIn, DecisionWebhookIn, DecisionWebhookOut

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit-bot")

_explainer: Explainer | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    global _explainer
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        _explainer = Explainer(artifact)
        logger.info("Modelo auditado carregado (%s).", _explainer.method)
    else:
        logger.warning(
            "Nenhum modelo encontrado em %s. Rode `python ml/generate_and_train.py` "
            "para gerar o modelo de demonstracao antes de enviar decisoes.",
            MODEL_PATH,
        )
    yield


app = FastAPI(
    title="Audit-Bot",
    description=(
        "Middleware de governanca de IA e mitigacao de vies algoritmico "
        "(LGPD Art. 6 IX / Art. 20 / Art. 50, CDC Art. 6 III / Art. 39)."
    ),
    version="0.1.0-mvp",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _row_to_transaction_dict(row) -> dict[str, Any]:
    return {
        "transaction_id": row["transaction_id"],
        "received_at": row["received_at"],
        "source_system": row["source_system"],
        "features_json": row["features_json"],
        "protected_attributes_json": row["protected_attributes_json"],
        "prediction": row["prediction"],
        "score": row["score"],
        "explanation_json": row["explanation_json"],
        "explanation_text": row["explanation_text"],
        "explanation_method": row["explanation_method"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _explainer is not None,
        "explanation_method": _explainer.method if _explainer else None,
        "transactions_audited": db.count_transactions(),
    }


# ---------------------------------------------------------------------------
# RF01 + RF02 + RF04 + RF03: ingestao passiva via webhook
# ---------------------------------------------------------------------------
@app.post("/webhook/decisions", response_model=DecisionWebhookOut)
def receive_decision(payload: DecisionWebhookIn) -> DecisionWebhookOut:
    if _explainer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo de explicabilidade nao carregado. Rode "
                "`python ml/generate_and_train.py` no servidor do Audit-Bot."
            ),
        )

    if db.get_transaction(payload.transaction_id) is not None:
        raise HTTPException(
            status_code=409,
            detail="transaction_id ja auditado (log e append-only, sem sobrescrita).",
        )

    missing = [c for c in _explainer.feature_columns if c not in payload.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Faltam features esperadas pelo modelo auditado: {missing}",
        )

    # RF04: explicabilidade (SHAP ou aproximacao linear)
    weights, explanation_text = _explainer.explain(payload.features)

    # RNF03: grava o registro imutavel (append-only)
    db.insert_transaction(
        transaction_id=payload.transaction_id,
        source_system=payload.source_system,
        features=payload.features,
        protected_attributes=payload.protected_attributes,
        prediction=payload.prediction,
        score=payload.score,
        explanation=weights,
        explanation_text=explanation_text,
        explanation_method=_explainer.method,
    )

    # RF02 + RF03: recalcula disparidade e dispara alertas se necessario
    alerts_triggered: list[dict[str, Any]] = []
    if payload.protected_attributes:
        alerts_triggered = fairness.run_fairness_checks(payload.transaction_id)
        for a in alerts_triggered:
            logger.warning("ALERTA DE GOVERNANCA: %s", a["message"])

    return DecisionWebhookOut(
        transaction_id=payload.transaction_id,
        prediction=payload.prediction,
        score=payload.score,
        explanation_method=_explainer.method,
        explanation=weights,
        explanation_text=explanation_text,
        alerts_triggered=alerts_triggered,
    )


# ---------------------------------------------------------------------------
# Consulta de transacoes / explicacao individual (perfil Operador)
# ---------------------------------------------------------------------------
@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str) -> dict[str, Any]:
    row = db.get_transaction(transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Transacao nao encontrada.")
    data = _row_to_transaction_dict(row)
    data["features"] = json.loads(data.pop("features_json"))
    data["protected_attributes"] = json.loads(data.pop("protected_attributes_json"))
    data["explanation"] = json.loads(data.pop("explanation_json"))
    return data


@app.get("/transactions/{transaction_id}/report")
def get_transaction_report(transaction_id: str) -> Response:
    row = db.get_transaction(transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Transacao nao encontrada.")

    protected_attrs = json.loads(row["protected_attributes_json"])
    related_alerts: list[dict[str, Any]] = []
    for attribute, value in protected_attrs.items():
        for alert_row in db.list_alerts(limit=200):
            if (
                alert_row["protected_attribute"] == attribute
                and alert_row["group_value"] == str(value)
            ):
                related_alerts.append(dict(alert_row))

    pdf_bytes = build_transaction_report(_row_to_transaction_dict(row), related_alerts)
    headers = {
        "Content-Disposition": f'attachment; filename="relatorio_lgpd_{transaction_id}.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


# ---------------------------------------------------------------------------
# Painel de Governanca (perfil DPO / Engenheiro de ML) -- RF03
# ---------------------------------------------------------------------------
@app.get("/governance/transactions")
def governance_transactions(limit: int = 50) -> list[dict[str, Any]]:
    rows = db.list_transactions(limit=limit)
    out = []
    for row in rows:
        d = _row_to_transaction_dict(row)
        d["features"] = json.loads(d.pop("features_json"))
        d["protected_attributes"] = json.loads(d.pop("protected_attributes_json"))
        d["explanation"] = json.loads(d.pop("explanation_json"))
        out.append(d)
    return out


@app.get("/governance/alerts")
def governance_alerts(limit: int = 50) -> list[dict[str, Any]]:
    rows = db.list_alerts(limit=limit)
    out = []
    for row in rows:
        d = dict(row)
        d["sample_transaction_ids"] = json.loads(d.pop("sample_transaction_ids_json"))
        d["reviews"] = [dict(r) for r in db.list_alert_reviews(row["id"])]
        d["status"] = "revisado" if d["reviews"] else "aberto"
        out.append(d)
    return out


@app.post("/governance/alerts/{alert_id}/review")
def review_alert(alert_id: int, payload: AlertReviewIn) -> dict[str, Any]:
    if db.get_alert(alert_id) is None:
        raise HTTPException(status_code=404, detail="Alerta nao encontrado.")
    review_id = db.insert_alert_review(
        alert_id=alert_id, reviewer=payload.reviewer, note=payload.note
    )
    return {"review_id": review_id, "alert_id": alert_id, "status": "revisado"}


@app.get("/governance/summary")
def governance_summary() -> dict[str, Any]:
    attributes = db.list_distinct_protected_attributes()
    summary = {}
    for attribute in attributes:
        stats = db.get_group_stats(attribute)
        summary[attribute] = [
            {
                "group_value": row["group_value"],
                "n": row["n"],
                "favorable_rate": row["favorable_rate"],
            }
            for row in stats
            if row["group_value"] is not None
        ]
    return {
        "total_transactions": db.count_transactions(),
        "total_alerts": len(db.list_alerts(limit=10_000)),
        "group_stats": summary,
    }


# ---------------------------------------------------------------------------
# Painel de Governanca (frontend estatico simples)
# ---------------------------------------------------------------------------
_dashboard_dir = BASE_DIR / "dashboard"
if _dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
