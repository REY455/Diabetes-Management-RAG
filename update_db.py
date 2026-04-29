import sqlite3

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(conversations)")
for col in cursor.fetchall():
    print(col)

conn.close()