import sqlite3
import os
from datetime import datetime

DB = 'data.db'
username = 'smoke_test_user'

conn = sqlite3.connect(DB)
cur = conn.cursor()
# Ensure trash table exists (in case init_db wasn't run)
cur.execute('''
CREATE TABLE IF NOT EXISTS history_trash (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    orig_id INTEGER,
    username TEXT,
    text TEXT,
    created_at TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

print('DB file exists:', os.path.exists(DB))

# Insert a test history entry
text = 'smoke test entry ' + datetime.now().isoformat()
cur.execute("INSERT INTO history (username, text, created_at) VALUES (?, ?, datetime('now'))", (username, text))
conn.commit()
new_id = cur.lastrowid
print('Inserted history id', new_id)

# Move to trash (simulate delete_history)
cur.execute('SELECT id, username, text, created_at FROM history WHERE id = ? AND username = ?', (new_id, username))
row = cur.fetchone()
if not row:
    print('Row not found for deletion')
else:
    orig_id, uname, ttext, created_at = row
    cur.execute('INSERT INTO history_trash (orig_id, username, text, created_at, deleted_at) VALUES (?, ?, ?, ?, datetime("now"))', (orig_id, uname, ttext, created_at))
    trash_id = cur.lastrowid
    cur.execute('DELETE FROM history WHERE id = ? AND username = ?', (new_id, username))
    conn.commit()
    print('Moved to trash id', trash_id)

# Restore last (simulate /restore-last)
cur.execute('SELECT id, orig_id, username, text, created_at FROM history_trash WHERE username = ? ORDER BY deleted_at DESC LIMIT 1', (username,))
row = cur.fetchone()
if not row:
    print('No trashed row to restore')
else:
    trash_id, orig_id, uname, ttext, created_at = row
    cur.execute('INSERT INTO history (username, text, created_at) VALUES (?, ?, ?)', (uname, ttext, created_at))
    new_restored_id = cur.lastrowid
    cur.execute('DELETE FROM history_trash WHERE id = ? AND username = ?', (trash_id, username))
    conn.commit()
    print('Restored to history id', new_restored_id)

# Cleanup: remove restored entry
cur.execute('DELETE FROM history WHERE id = ? AND username = ?', (new_restored_id, username))
conn.commit()
print('Cleaned up restored entry')
conn.close()
print('Smoke test completed')
