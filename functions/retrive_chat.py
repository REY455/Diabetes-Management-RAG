import sqlite3

def get_chat_history(limit=5):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT user_query, response 
    FROM conversations 
    ORDER BY id DESC 
    LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return rows[::-1]  # oldest → newest