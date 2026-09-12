"""
Simula o "sistema de IA/ML cliente" (ex: uma fintech de credito) enviando
copias passivas das suas decisoes ao Audit-Bot via webhook (RF01),
exatamente como descrito no RNF01: o Audit-Bot fica desacoplado e so
recebe uma copia, nunca intercepta o fluxo real.

Uso:
    python scripts/simulate_traffic.py --n 300 --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

import httpx
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.generate_and_train import FEATURE_COLUMNS, generate_dataset  # noqa: E402


def make_transaction_id() -> str:
    """Gera um identificador pseudonimo (UUID) -- nunca um dado pessoal
    direto, conforme RNF02."""
    return uuid.uuid4().hex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="numero de decisoes simuladas")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "ml" / "model.joblib"),
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    pipeline = artifact["model"]

    df = generate_dataset(n_samples=args.n)
    df["prediction"] = pipeline.predict(df[FEATURE_COLUMNS])
    df["score"] = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]

    sent, alerts_seen = 0, 0
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for _, row in df.iterrows():
            payload = {
                "transaction_id": make_transaction_id(),
                "source_system": "fintech-credito-demo",
                "features": {col: float(row[col]) for col in FEATURE_COLUMNS},
                "protected_attributes": {
                    "genero": row["genero"],
                    "regiao": row["regiao"],
                },
                "prediction": int(row["prediction"]),
                "score": float(row["score"]),
            }
            resp = client.post("/webhook/decisions", json=payload)
            if resp.status_code != 200:
                print(f"[erro {resp.status_code}] {resp.text[:200]}")
                continue
            sent += 1
            body = resp.json()
            if body["alerts_triggered"]:
                alerts_seen += len(body["alerts_triggered"])
                for a in body["alerts_triggered"]:
                    print(f"[ALERTA] {a['message']}")

    print(f"\nEnviadas {sent}/{len(df)} decisoes. Alertas novos disparados: {alerts_seen}.")
    print(f"Painel de governanca: {args.url}/dashboard/")


if __name__ == "__main__":
    main()
