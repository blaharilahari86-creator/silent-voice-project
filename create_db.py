import sqlite3

conn = sqlite3.connect("data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    text TEXT,
    FOREIGN KEY(username) REFERENCES users(username)
)
""")

cursor.execute("PRAGMA table_info(history)")
columns = [row[1] for row in cursor.fetchall()]
if "username" not in columns:
    cursor.execute("ALTER TABLE history ADD COLUMN username TEXT")
    cursor.execute("UPDATE history SET username = 'guest' WHERE username IS NULL")

conn.commit()
conn.close()

print("Database created or migrated")