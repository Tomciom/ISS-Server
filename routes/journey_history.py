from flask import Blueprint, render_template, session
import sqlite3

bp = Blueprint('journey_history', __name__)

def get_user_id(username):
    conn = sqlite3.connect('Users.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cur.fetchone()
    conn.close()
    return result['id'] if result else None

def get_user_mac_addresses(user_id):
    conn = sqlite3.connect('Users.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT mac_address FROM user_boards WHERE user_id = ?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [row['mac_address'] for row in rows] if rows else []

def get_board_name(mac_address):
    conn = sqlite3.connect('Users.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT board_name FROM user_boards WHERE mac_address = ?", (mac_address,))
    result = cur.fetchone()
    conn.close()
    return result['board_name'] if result else "Nieznana płyta"

def get_all_journeys(username):
    user_id = get_user_id(username)
    if not user_id:
        return []
    
    mac_addresses = get_user_mac_addresses(user_id)
    if not mac_addresses:
        return []

    placeholders = ','.join('?' for _ in mac_addresses)
    query = f"""
        SELECT id, start_time, end_time, mac_address 
        FROM journeys 
        WHERE mac_address IN ({placeholders}) 
        ORDER BY start_time DESC
    """

    conn = sqlite3.connect('journeys.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, mac_addresses)
    journeys_raw = cur.fetchall()
    conn.close()

    journeys = []
    for row in journeys_raw:
        journey_id = row['id']
        start_time = row['start_time']
        end_time = row['end_time']
        mac_address = row['mac_address']
        board_name = get_board_name(mac_address)
        journeys.append({
            'id': journey_id,
            'board_name': board_name,
            'start_time': start_time,
            'end_time': end_time
        })

    return journeys

@bp.route('/journey_history', methods=['GET'])
def journey_history():
    username = session.get('username')
    journeys = get_all_journeys(username)
    return render_template('journey_history.html', journeys=journeys, username=username)
