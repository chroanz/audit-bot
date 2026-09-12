from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import FORBIDDEN_FIELDS


class DecisionWebhookIn(BaseModel):
    """Payload que o sistema de IA/ML cliente envia ao Audit-Bot (RF01).

    E uma copia PASSIVA da decisao ja tomada: o Audit-Bot audita, mas
    nao intercepta nem bloqueia o fluxo original (RNF01 - desacoplamento
    critico). O transaction_id deve ser um hash ou UUID gerado pelo
    sistema cliente (RNF02 - nunca um dado pessoal direto).
    """

    transaction_id: str = Field(..., min_length=8, max_length=200)
    source_system: Optional[str] = Field(default=None, max_length=100)
    features: dict[str, float] = Field(
        ..., description="Variaveis legitimas usadas pelo modelo (nao sensiveis)."
    )
    protected_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Atributos protegidos apenas para fins de auditoria de equidade.",
    )
    prediction: int = Field(..., ge=0, le=1, description="0 = negado, 1 = aprovado")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("features", "protected_attributes")
    @classmethod
    def reject_pii(cls, value: dict[str, Any]) -> dict[str, Any]:
        offending = FORBIDDEN_FIELDS.intersection({k.lower() for k in value.keys()})
        if offending:
            raise ValueError(
                "Campos de dado pessoal direto nao sao permitidos "
                f"(RNF02 - privacidade by design): {sorted(offending)}"
            )
        return value

    @field_validator("transaction_id")
    @classmethod
    def transaction_id_must_look_like_hash_or_uuid(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[a-fA-F0-9-]{8,200}", value):
            raise ValueError(
                "transaction_id deve ser um hash/UUID (apenas hexadecimal e hifens), "
                "nunca um identificador pessoal (RNF02)."
            )
        return value


class DecisionWebhookOut(BaseModel):
    transaction_id: str
    prediction: int
    score: Optional[float]
    explanation_method: str
    explanation: dict[str, float]
    explanation_text: str
    alerts_triggered: list[dict[str, Any]] = Field(default_factory=list)


class AlertReviewIn(BaseModel):
    reviewer: str = Field(..., min_length=2, max_length=150)
    note: str = Field(..., min_length=2, max_length=2000)
