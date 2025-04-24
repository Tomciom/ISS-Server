from flask import Blueprint, render_template, flash, redirect, url_for, session, request
import sqlite3

bp = Blueprint('login', __name__)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('Users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = username
            flash('Zalogowano pomyślnie!')
            return redirect(url_for('home.home'))
        else:
            flash('Błędna nazwa użytkownika lub hasło!')
            return redirect(url_for('login.login'))

    return render_template('login.html')


@bp.route('/logout')
def logout():
    session.pop('username', None)
    flash('Wylogowano pomyślnie!')
    return redirect(url_for('home.home'))