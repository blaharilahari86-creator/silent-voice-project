import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()

# Check schema
c.execute("PRAGMA table_info(history)")
cols = [row[1] for row in c.fetchall()]
print(f'History table columns: {cols}\n')

c.execute('SELECT COUNT(*) FROM history')
total = c.fetchone()[0]

# Query with all 4 columns
try:
    c.execute('SELECT id, username, text, created_at FROM history ORDER BY id DESC LIMIT 10')
    recent = c.fetchall()
    print(f'Total history entries: {total}')
    print('\nMost recent 10 entries (with timestamps):')
    for row in recent:
        text_preview = row[2][:50] if row[2] else '(empty)'
        timestamp = row[3] if row[3] else '(no timestamp)'
        print(f'  ID {row[0]:4d} | {row[1]:20s} | {text_preview:50s} | {timestamp}')
except Exception as e:
    print(f'Error: {e}')
    print('Trying without created_at...')
    c.execute('SELECT id, username, text FROM history ORDER BY id DESC LIMIT 5')
    recent = c.fetchall()
    print(f'Total history entries: {total}')
    print('\nMost recent 5 entries:')
    for row in recent:
        text_preview = row[2][:50] if row[2] else '(empty)'
        print(f'  ID {row[0]:4d} | {row[1]:20s} | {text_preview}')

conn.close()
