# routes/device_data.py
from flask import Blueprint, render_template, session, redirect, url_for, abort, g
import sqlite3
import logging
import pandas as pd
from io import BytesIO
from flask import make_response


# Assuming DB_PATH is available or imported (it's in app.py)
DB_PATH = 'measurements.db'

# Zmieniona nazwa blueprintu
bp = Blueprint('device_data', __name__)

# Helper to get database connection (standard Flask/SQLite pattern)
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row # Allows accessing columns by name
    return db

# Close database connection at the end of the request
@bp.teardown_request
def close_db(error):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# Zmieniona nazwa endpointu i URL
@bp.route('/device_data/<mac_address>', methods=['GET'])
def device_data_by_mac(mac_address):
    # 1. Sprawdzenie czy użytkownik jest zalogowany (pierwsza linia obrony)
    if 'username' not in session:
        logging.warning(f"Attempted access to /device_data/{mac_address} by unauthenticated user.")
        return redirect(url_for('login.login'))

    # 2. Pobranie nazwy użytkownika z sesji
    username = session['username']
    conn = get_db() # Użyj helpera do połączenia z bazą
    cur = conn.cursor()

    measurements = []

    try:
        # 3. Sprawdzenie CZY MAC ADRES PRZYPISANY JEST DO ZALOGOWANEGO UŻYTKOWNIKA
        # Znajdź user_id dla zalogowanego użytkownika
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_row = cur.fetchone()
        if not user_row:
             # To nie powinno się zdarzyć, jeśli użytkownik jest zalogowany, ale dla bezpieczeństwa
             logging.error(f"Logged in user '{username}' not found in database during device data access.")
             abort(404, description="Użytkownik nie znaleziony.")

        user_id = user_row['id']

        # Sprawdź, czy istnieje wpis w user_boards dla tego user_id i tego mac_address
        cur.execute("SELECT COUNT(*) FROM user_boards WHERE user_id = ? AND mac_address = ?", (user_id, mac_address))
        board_count = cur.fetchone()[0]

        if board_count == 0:
             # Urządzenie nie przypisane do tego użytkownika lub nie istnieje
             logging.warning(f"Access denied: User '{username}' attempted to access unauthorized MAC '{mac_address}'")
             abort(403, description="To urządzenie nie jest przypisane do Twojego konta lub nie istnieje w systemie.") # Zwróć 403 Forbidden


        # Jeśli dotarliśmy tutaj, użytkownik jest zalogowany I posiada to urządzenie
        # 4. Pobranie wszystkich pomiarów dla tego mac_address (istniejąca logika)
        cur.execute(
            "SELECT * FROM measurements WHERE mac_address = ? ORDER BY server_timestamp",
            (mac_address,)
        )
        all_measurements = cur.fetchall()

        # Konwersja wierszy na słowniki
        measurements = [dict(row) for row in all_measurements]

    except sqlite3.Error as e:
        logging.error(f"Database error in device_data_by_mac during authorization or fetch: {e}")
        # Użyj abort z konkretnym kodem błędu dla błędów bazy danych
        abort(500, description="Błąd bazy danych podczas pobierania danych urządzenia.")

    # Połączenie z bazą danych zostanie automatycznie zamknięte przez bp.teardown_appcontext

    # 5. Renderowanie zaadaptowanego szablonu (istniejąca logika)
    # Przekazujemy mac_address, username i pobrane pomiary do szablonu
    return render_template(
        'device_data.html', # Zaktualizowana nazwa szablonu
        mac_address=mac_address,
        username=username, # Przekazujemy username do szablonu np. do paska bocznego
        measurements=measurements,
    )

# Dodanie endpointu do pobierania CSV dla konkretnego MAC adresu z zabezpieczeniem
@bp.route('/device_data/<mac_address>/download', methods=['GET'])
def download_csv(mac_address):
    # 1. Sprawdzenie czy użytkownik jest zalogowany
    if 'username' not in session:
        logging.warning(f"Attempted CSV download for {mac_address} by unauthenticated user.")
        return redirect(url_for('login.login'))

    # 2. Pobranie nazwy użytkownika z sesji
    username = session['username']
    conn = get_db()
    cur = conn.cursor()

    try:
        # 3. Sprawdzenie CZY MAC ADRES PRZYPISANY JEST DO ZALOGOWANEGO UŻYTKOWNIKA
        cur.execute(
            "SELECT COUNT(*) FROM user_boards WHERE user_id = (SELECT id FROM users WHERE username = ?) AND mac_address = ?",
            (username, mac_address)
        )
        board_count = cur.fetchone()[0]

        if board_count == 0:
             # Urządzenie nie przypisane do tego użytkownika lub nie istnieje
             logging.warning(f"Access denied: User '{username}' attempted to download data for unauthorized MAC '{mac_address}'")
             abort(403, description="To urządzenie nie jest przypisane do Twojego konta lub nie istnieje w systemie.") # Zwróć 403 Forbidden

        # Jeśli dotarliśmy tutaj, użytkownik jest zalogowany I posiada to urządzenie
        # 4. Pobranie wszystkich pomiarów dla tego mac_address (istniejąca logika)
        cur.execute(
            "SELECT server_timestamp, temperature, pressure, humidity, sunshine, wind_speed, wind_direction FROM measurements WHERE mac_address = ? ORDER BY server_timestamp",
            (mac_address,)
        )
        all_measurements = cur.fetchall()

        if not all_measurements:
            # Brak danych pomiarowych DLA TEGO urządzenia, ale użytkownik je posiada
            return "Brak danych do pobrania dla tego urządzenia.", 404 # Zwróć 404 Not Found (danych)

        # 5. Konwersja danych na CSV i zwrócenie odpowiedzi (istniejąca logika)
        df = pd.DataFrame(all_measurements, columns=[
            'Timestamp', 'Temperatura (°C)', 'Ciśnienie (hPa)', 'Wilgotność (%)',
            'Wykryto Światło (Surowa Wartość)', 'Prędkość Wiatru (m/s)', 'Kierunek Wiatru'
        ])

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig') # Użyj utf-8-sig dla BOM (lepsza kompatybilność z Excel)
        csv_buffer.seek(0)

        response = make_response(csv_buffer.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=device_data_{mac_address.replace(":", "-")}.csv'
        response.headers['Content-type'] = 'text/csv'

        return response

    except sqlite3.Error as e:
        logging.error(f"Database error in download_csv during authorization or fetch: {e}")
        abort(500, description="Błąd bazy danych podczas generowania pliku CSV.")

    # Połączenie z bazą danych zostanie automatycznie zamknięte przez bp.teardown_appcontext