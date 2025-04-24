from flask import Blueprint, render_template, request, flash, redirect, url_for
import sqlite3

bp = Blueprint('register', __name__)

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Hasła nie są zgodne! Spróbuj ponownie.')
            return redirect(url_for('register.register'))
        try:
            conn = sqlite3.connect('Users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            flash('Użytkownik o podanej nazwie już istnieje!')
            return redirect(url_for('register.register'))

        flash('Rejestracja zakończona sukcesem!')
        return redirect(url_for('home.home'))
    return render_template('register.html')