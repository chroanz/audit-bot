"""
RF05 (Relatorio de Revisao para a LGPD): gera um PDF com os motivos da
decisao automatizada para atender pedidos de revisao do Art. 20 da LGPD,
incluindo o contexto de equidade (se a transacao esta associada a algum
alerta de disparidade em aberto).
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

STYLES = getSampleStyleSheet()
STYLES.add(
    ParagraphStyle(name="AuditTitle", fontSize=16, spaceAfter=12, leading=20)
)
STYLES.add(
    ParagraphStyle(name="AuditSection", fontSize=12, spaceBefore=14, spaceAfter=6, leading=15)
)


def build_transaction_report(transaction: dict[str, Any], related_alerts: list[dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"Relatorio LGPD Art. 20 - {transaction['transaction_id']}",
    )

    story = []
    story.append(Paragraph("Relatório de Revisão de Decisão Automatizada", STYLES["AuditTitle"]))
    story.append(
        Paragraph(
            "Documento gerado pelo Audit-Bot em atendimento ao direito de revisão de "
            "decisões automatizadas previsto no <b>Art. 20 da Lei nº 13.709/2018 (LGPD)</b>.",
            STYLES["Normal"],
        )
    )

    story.append(Paragraph("1. Identificação da Transação", STYLES["AuditSection"]))
    meta_table_data = [
        ["ID da transação (hash/UUID)", transaction["transaction_id"]],
        ["Recebido em (UTC)", transaction["received_at"]],
        ["Sistema de origem", transaction.get("source_system") or "não informado"],
        [
            "Decisão automatizada",
            "APROVADO" if transaction["prediction"] == 1 else "NEGADO",
        ],
        [
            "Confiança do modelo (score)",
            f"{transaction['score']:.2%}" if transaction.get("score") is not None else "não informado",
        ],
        ["Método de explicabilidade", transaction["explanation_method"]],
    ]
    meta_table = Table(meta_table_data, colWidths=[7 * cm, 9 * cm])
    meta_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(meta_table)

    story.append(Paragraph("2. Motivos da Decisão (Explicabilidade)", STYLES["AuditSection"]))
    story.append(Paragraph(transaction["explanation_text"], STYLES["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    explanation = json.loads(transaction["explanation_json"])
    weight_rows = [["Variável", "Peso na decisão"]]
    for name, weight in sorted(explanation.items(), key=lambda kv: abs(kv[1]), reverse=True):
        weight_rows.append([name, f"{weight:+.4f}"])
    weight_table = Table(weight_rows, colWidths=[9 * cm, 7 * cm])
    weight_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(weight_table)

    story.append(Paragraph("3. Dados Utilizados na Decisão", STYLES["AuditSection"]))
    features = json.loads(transaction["features_json"])
    feature_rows = [["Variável", "Valor"]] + [[k, str(v)] for k, v in features.items()]
    feature_table = Table(feature_rows, colWidths=[9 * cm, 7 * cm])
    feature_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(feature_table)

    story.append(Paragraph("4. Contexto de Equidade Algorítmica", STYLES["AuditSection"]))
    if related_alerts:
        story.append(
            Paragraph(
                "Esta transação está associada a um ou mais alertas de possível "
                "disparidade estatística (viés) ativos no momento da consulta:",
                STYLES["Normal"],
            )
        )
        for alert in related_alerts:
            story.append(Paragraph(f"• {alert['message']}", STYLES["Normal"]))
    else:
        story.append(
            Paragraph(
                "Nenhum alerta de disparidade estatística associado a esta transação "
                "estava em aberto no momento da geração deste relatório.",
                STYLES["Normal"],
            )
        )

    story.append(Paragraph("5. Base Legal", STYLES["AuditSection"]))
    story.append(
        Paragraph(
            "LGPD Art. 6º, IX (não discriminação); Art. 20 (revisão de decisões "
            "automatizadas); Art. 50 (boas práticas e governança); "
            "CDC Art. 6º, III (direito à informação).",
            STYLES["Normal"],
        )
    )

    doc.build(story)
    return buffer.getvalue()
