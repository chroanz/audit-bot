from pathlib import Path
import sqlite3
import tempfile

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / "results/auditoria_german_credit.db"

assert ORIGINAL.exists(), "Banco experimental não encontrado"

tabelas = ["transacoes", "alertas", "explicacoes"]

with tempfile.TemporaryDirectory() as pasta:
    copia = Path(pasta) / "auditoria_teste.db"

    # Backup consistente, sem modificar o original
    with sqlite3.connect(ORIGINAL) as origem:
        with sqlite3.connect(copia) as destino:
            origem.backup(destino)

    with sqlite3.connect(copia) as conn:
        for tabela in tabelas:
            registro = conn.execute(f"SELECT rowid FROM {tabela} LIMIT 1").fetchone()

            assert registro is not None, f"Tabela {tabela} não contém registros"

            rowid = registro[0]

            print(f"\n=== TABELA: {tabela} ===")

            # Verificar proteção contra UPDATE
            try:
                conn.execute(
                    f"UPDATE {tabela} SET rowid = rowid WHERE rowid = ?", (rowid,)
                )
            except sqlite3.DatabaseError as erro:
                print("UPDATE bloqueado:", erro)
            else:
                print("ATENÇÃO: UPDATE não foi bloqueado")

            conn.rollback()

            # Verificar proteção contra DELETE
            try:
                conn.execute(f"DELETE FROM {tabela} WHERE rowid = ?", (rowid,))
            except sqlite3.DatabaseError as erro:
                print("DELETE bloqueado:", erro)
            else:
                print("ATENÇÃO: DELETE não foi bloqueado")

            conn.rollback()

print("\nTeste concluído em cópia temporária.")
print("Banco original preservado.")
