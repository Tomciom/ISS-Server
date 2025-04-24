from flask import Blueprint, render_template, request, redirect, url_for, session
import sqlite3

bp = Blueprint('boards', __name__)

def delete_related_journey_data(mac_address):
    conn = sqlite3.connect('journeys.db')
    c = conn.cursor()

    c.execute('DELETE FROM temperature_pressure WHERE mac_address = ?', (mac_address,))

    c.execute('DELETE FROM fire_detection WHERE mac_address = ?', (mac_address,))

    c.execute('DELETE FROM rotation_acceleration WHERE mac_address = ?', (mac_address,))

    c.execute('DELETE FROM journeys WHERE mac_address = ?', (mac_address,))
    
    conn.commit()
    conn.close()

def delete_board_and_related_data(board_id, mac_address):
    conn = sqlite3.connect('Users.db')
    c = conn.cursor()
    c.execute('DELETE FROM user_boards WHERE id = ?', (board_id,))
    conn.commit()
    conn.close()

    delete_related_journey_data(mac_address)

def get_user_boards(username):
    conn = sqlite3.connect('Users.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''SELECT user_boards.id, user_boards.mac_address, user_boards.board_name 
                 FROM user_boards 
                 INNER JOIN users ON user_boards.user_id = users.id 
                 WHERE users.username = ? 
                 ORDER BY user_boards.id ASC''', (username,))
    boards = c.fetchall()
    conn.close()
    return boards

def update_board_name(board_id, new_name):
    conn = sqlite3.connect('Users.db')
    c = conn.cursor()
    c.execute('UPDATE user_boards SET board_name = ? WHERE id = ?', (new_name, board_id))
    conn.commit()
    conn.close()

def toggle_board_usage(board_id, in_use):
    conn = sqlite3.connect('Users.db')
    c = conn.cursor()
    c.execute('UPDATE user_boards SET is_in_use = ? WHERE id = ?', (in_use, board_id))
    conn.commit()
    conn.close()

@bp.route('/<username>/boards', methods=['GET', 'POST'])
def boards(username):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('home.home'))
    
    if request.method == 'POST':
        if 'reset' in request.form:
            board_id = request.form.get('board_id')
            if board_id:
                toggle_board_usage(board_id, 0)
            return redirect(url_for('boards.boards', username=username))

    if request.method == 'POST':
        if 'delete' in request.form:
            board_id = request.form.get('board_id')
            mac_address = request.form.get('mac_address')
            if board_id and mac_address:
                delete_board_and_related_data(board_id, mac_address)
            return redirect(url_for('boards.boards', username=username))

        board_id = request.form.get('board_id')
        new_name = request.form.get('new_name')
        in_use = request.form.get('is_in_use')
        if board_id:
            if new_name:
                update_board_name(board_id, new_name)
            if in_use is not None:
                toggle_board_usage(board_id, int(in_use))
        
        return redirect(url_for('boards.boards', username=username))

    boards = get_user_boards(username)
    return render_template('boards.html', boards=boards, username=username)
