import os
import sqlite3
from app import app

client = app.test_client()

def remove_user(username):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE username = ?", (username,))
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

username = 'e2e_test_user'
password = 'testpass'

# ensure clean state
remove_user(username)

print('--- Register ---')
res = client.post('/register', json={'username': username, 'password': password})
print(res.status_code, res.get_json())

print('--- Login ---')
res = client.post('/login', json={'username': username, 'password': password})
print(res.status_code, res.get_json())

print('--- Unauthenticated dashboard redirect check ---')
c2 = app.test_client()
r = c2.get('/', follow_redirects=False)
print('GET / ->', r.status_code, r.headers.get('Location'))

print('--- Speak (TTS) ---')
res = client.post('/speak', json={'text': 'hello world from e2e', 'language': 'en'})
print(res.status_code, res.get_json())
res_json = res.get_json() or {}
audio_url = res_json.get('audio')
exists = False
fname = None
if audio_url:
    fname = audio_url.split('/')[-1]
    exists = os.path.exists(os.path.join('static', fname))
print('audio_url=', audio_url, 'exists=', exists)

print('--- Save text (via /save) ---')
res = client.post('/save', json={'text': 'saved by e2e test'})
print(res.status_code, res.get_json())

print('--- DB check last history for user ---')
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('SELECT id, username, text FROM history WHERE username = ? ORDER BY id DESC LIMIT 1', (username,))
row = c.fetchone()
conn.close()
print('last history row:', row)

print('--- Cleanup ---')
if fname and os.path.exists(os.path.join('static', fname)):
    try:
        os.remove(os.path.join('static', fname))
        print('removed audio file', fname)
    except Exception as e:
        print('failed to remove audio file', e)

remove_user(username)
print('cleanup DB done')
print('E2E test finished')
