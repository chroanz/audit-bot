from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def gerar_relatorio(transacao, explicacao, equidade, alerta):
    buffer = BytesIO()
    documento = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    estilos = getSampleStyleSheet()
    estilos.add(
        ParagraphStyle(
            name="TituloAuditoria",
            parent=estilos["Heading1"],
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )

    elementos = []

    def texto(valor):
        return escape(str(valor))

    def paragrafo(conteudo):
        elementos.append(Paragraph(conteudo, estilos["Normal"]))
        elementos.append(Spacer(1, 0.3 * cm))

    def secao(titulo):
        elementos.append(Paragraph(titulo, estilos["Heading2"]))

    def tabela(linhas):
        dados = [
            [Paragraph(texto(celula), estilos["Normal"]) for celula in linha]
            for linha in linhas
        ]

        t = Table(
            dados,
            colWidths=[6 * cm, 10 * cm],
            repeatRows=1,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.grey),
                ]
            )
        )
        elementos.append(t)
        elementos.append(Spacer(1, 0.5 * cm))

    elementos.append(
        Paragraph(
            "AUDIT-BOT — RELATÓRIO TÉCNICO DE AUDITORIA",
            estilos["TituloAuditoria"],
        )
    )

    paragrafo(
        "Simulação acadêmica com German Credit Data. "
        "Documento de apoio à investigação e revisão humana."
    )

    secao("1. Identificação da transação")

    tabela(
        [
            ["Campo", "Valor"],
            ["UUID", transacao["transaction_id"]],
            ["Recebimento (UTC)", transacao["received_at"]],
            ["Grupo de idade", transacao["grupo"]],
            ["Modelo SHA-256", transacao["model_id"]],
            ["Decisão", str(transacao["prediction"])],
            ["Probabilidade de bom risco", f"{transacao['score']:.4%}"],
        ]
    )

    secao("2. Explicação individual SHAP")

    tabela(
        [
            ["Campo", "Valor"],
            ["Método", explicacao["metodo"]],
            ["Escala", explicacao["escala"]],
            ["Valor de referência", f"{explicacao['valor_referencia']:.6f}"],
            ["Escore linear", f"{explicacao['escore_linear']:.6f}"],
            ["Erro de reconstrução", f"{explicacao['erro_reconstrucao']:.10f}"],
        ]
    )

    contribuicoes = sorted(
        explicacao["contribuicoes"].items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    tabela(
        [["Atributo", "Contribuição (log-odds)"]]
        + [[nome, f"{valor:+.6f}"] for nome, valor in contribuicoes]
    )

    secao("3. Equidade do lote experimental")

    tabela(
        [
            ["Indicador", "Resultado"],
            ["Transações", equidade["total_transactions"]],
            ["DPD", f"{equidade['DPD']:.6f}"],
            [
                "DI direcional",
                f"{equidade['DI']:.6f}" if equidade["DI"] is not None else "Indefinida",
            ],
            ["Limite operacional de DPD", f"{equidade['limiar_DPD']:.2f}"],
            ["Alerta de disparidade", "Sim" if equidade["alerta_DPD"] else "Não"],
        ]
    )

    secao("4. Alerta registrado")

    if alerta:
        paragrafo(f"Identificador do alerta: {alerta['id']}")
        paragrafo(texto(alerta["mensagem"]))
    else:
        paragrafo("Nenhum alerta registrado para este lote.")

    secao("5. Limitações e interpretação")

    paragrafo(
        "As contribuições SHAP descrevem o comportamento "
        "do classificador em log-odds, não representam "
        "percentuais de responsabilidade nem comprovam causalidade."
    )
    paragrafo(
        "A identificação de disparidades entre grupos exige "
        "investigação contextual e não constitui, isoladamente, "
        "comprovação de discriminação ilícita."
    )
    paragrafo(
        "Este relatório não substitui revisão humana, "
        "avaliação jurídica ou procedimentos organizacionais "
        "de governança e proteção de dados."
    )

    documento.build(elementos)

    return buffer.getvalue()
