import sqlite3

def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_query TEXT,
        response TEXT,
        model TEXT,

        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,

        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()