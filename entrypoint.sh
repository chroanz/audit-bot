#!/bin/sh
# Sobe o Audit-Bot e, se o log de auditoria estiver vazio (primeira
# execucao ou volume recem-criado), popula automaticamente com uma
# simulacao de trafego para que o painel de governanca ja abra pronto
# para demonstracao (alertas de disparidade incluidos).
set -e

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

trap 'echo "[entrypoint] encerrando..."; kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID"' TERM INT

echo "[entrypoint] aguardando o Audit-Bot ficar saudavel..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 0.5
done

TX_COUNT=$(curl -s http://localhost:8000/health | python3 -c "import json,sys; print(json.load(sys.stdin)['transactions_audited'])" 2>/dev/null || echo "0")

if [ "$TX_COUNT" = "0" ]; then
  echo "[entrypoint] log de auditoria vazio -> gerando dados de demonstracao (300 decisoes simuladas)..."
  python3 scripts/simulate_traffic.py --n 300 --url http://localhost:8000 || echo "[entrypoint] aviso: simulacao de demonstracao falhou, seguindo sem seed."
else
  echo "[entrypoint] log de auditoria ja possui $TX_COUNT transacoes -> pulando seed de demonstracao."
fi

echo "[entrypoint] Audit-Bot pronto em http://localhost:8000/dashboard/"
wait "$SERVER_PID"
