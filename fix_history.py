import sqlite3

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Update all NULL timestamps with current timestamp
cursor.execute("UPDATE history SET created_at = datetime('now') WHERE created_at IS NULL")
conn.commit()

# Verify the fix
cursor.execute('SELECT COUNT(*) FROM history WHERE created_at IS NULL')
null_count = cursor.fetchone()[0]
print(f'NULL timestamps remaining: {null_count}')

# Show recent entries
cursor.execute('SELECT id, username, text, created_at FROM history ORDER BY id DESC LIMIT 5')
print('\nMost recent entries:')
for row in cursor.fetchall():
    print(f'ID {row[0]} | {row[1]} | {row[2][:40]} | {row[3]}')

conn.close()
print('\nHistory timestamps fixed successfully!')
