import sqlite3

conn = sqlite3.connect("matches.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM matches")
conn.commit()
conn.close()

print("✅ Banco limpo com sucesso.")
