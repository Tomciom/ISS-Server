from flask import Blueprint, render_template, session, request, jsonify
import sqlite3

bp = Blueprint('home', __name__)

def send_code(username):
    conn = sqlite3.connect('measurements.db')
    c = conn.cursor()

    c.execute("SELECT pin FROM users WHERE username = ?", (username,))

    result = c.fetchone()
    
    if result:
        pin = result[0]
        print("Received configuration code:", pin)
    else:
        print("User not found.")
        pin = None
    
    conn.close()
    return pin



@bp.route('/', methods=['GET', 'POST'])
def home():
    username = session.get('username')
    if request.method == 'POST':
        newCode = request.json
        try:
            conn = sqlite3.connect('measurements.db')
            c = conn.cursor()
            c.execute("UPDATE users SET pin = ? WHERE username = ?", (newCode['code'], username))
            print("Received configuration code:", newCode['code'])
            conn.commit()
            conn.close()
            return jsonify({'message': 'Kod został zapisany!'}), 200
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'message': 'Wystąpił błąd podczas zapisywania kodu.'}), 500
    return render_template('home.html', username=username)