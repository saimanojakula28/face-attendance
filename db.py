import sqlite3
from pathlib import Path
from datetime import datetime, date

DB_PATH = Path("database.db")

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            att_date TEXT NOT NULL,
            att_time TEXT NOT NULL,
            UNIQUE(user_id, att_date)
        )
    """)
    conn.commit()
    conn.close()

def add_user(user_id: str, name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
        (user_id.strip(), name.strip(), datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()

def get_users():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, name, created_at FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def mark_attendance(user_id: str, name: str) -> bool:
    """Returns True if marked now, False if already marked today."""
    conn = get_conn()
    cur = conn.cursor()
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")
    try:
        cur.execute(
            "INSERT INTO attendance (user_id, name, att_date, att_time) VALUES (?, ?, ?, ?)",
            (user_id, name, today, now_time)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_attendance(date_filter: str | None = None):
    conn = get_conn()
    cur = conn.cursor()
    if date_filter:
        cur.execute(
            "SELECT user_id, name, att_date, att_time FROM attendance WHERE att_date=? ORDER BY att_time DESC",
            (date_filter,)
        )
    else:
        cur.execute(
            "SELECT user_id, name, att_date, att_time FROM attendance ORDER BY att_date DESC, att_time DESC"
        )
    rows = cur.fetchall()
    conn.close()
    return rows
