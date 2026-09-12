from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "audit_log.db"
MODEL_PATH = BASE_DIR / "ml" / "model.joblib"

# RF02 / disparate impact: regra dos 4/5 (80%) amplamente usada em
# fairness (EEOC four-fifths rule). Abaixo disso, dispara alerta.
DISPARATE_IMPACT_THRESHOLD = 0.8

# Numero minimo de decisoes por grupo antes de considerarmos a amostra
# estatisticamente relevante o suficiente para gerar um alerta.
MIN_GROUP_SAMPLE_SIZE = 10

# Campos que NUNCA devem ser aceitos no payload (RNF02 - privacidade by
# design: nenhum dado pessoal direto deve trafegar para o Audit-Bot).
FORBIDDEN_FIELDS = {
    "cpf",
    "rg",
    "nome",
    "nome_completo",
    "email",
    "telefone",
    "celular",
    "endereco",
    "passaporte",
    "cnh",
}
