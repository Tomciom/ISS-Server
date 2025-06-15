# routes/device_data.py
from flask import Blueprint, render_template, session, redirect, url_for, abort, g, jsonify, make_response
import sqlite3
import logging
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta


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

# --- NOWA FUNKCJA DO AGREGACJI DANYCH ---
def get_aggregated_data(mac_address, days=3):
    """
    Pobiera i agreguje dane pomiarowe dla danego MAC adresu z ostatnich X dni
    względem ostatniego zapisu w bazie danych.
    """
    conn = get_db()
    cur = conn.cursor()

    # --- NOWA LOGIKA: ZNAJDOWANIE OSTATNIEGO POMIARU ---
    # 1. Znajdź najnowszy timestamp dla tego urządzenia
    cur.execute("SELECT MAX(server_timestamp) FROM measurements WHERE mac_address = ?", (mac_address,))
    latest_timestamp_str = cur.fetchone()[0]

    # Jeśli nie ma żadnych pomiarów dla tego urządzenia, zwróć pustą listę
    if not latest_timestamp_str:
        return []

    # 2. Przekonwertuj string na obiekt datetime i oblicz zakres
    try:
        # SQLite zwraca string, więc musimy go sparsować
        # Format daty z SQLite to 'YYYY-MM-DD HH:MM:SS'
        latest_timestamp = datetime.strptime(latest_timestamp_str, '%Y-%m-%d %H:%M:%S')
        start_date = latest_timestamp - timedelta(days=days)
    except (ValueError, TypeError) as e:
        # Zabezpieczenie na wypadek problemów z formatem daty
        logging.error(f"Nie można sparsować daty '{latest_timestamp_str}': {e}")
        return []
    # --- KONIEC NOWEJ LOGIKI ---

    # Zapytanie SQL do agregacji danych (pozostaje prawie bez zmian,
    # ale teraz używa dynamicznego 'start_date' i dodatkowo 'end_date')
    query = """
        SELECT
            strftime('%Y-%m-%d %H', server_timestamp) || ':' || 
            printf('%02d', (strftime('%M', server_timestamp) / 30) * 30) as time_window,
            AVG(temperature) as avg_temp,
            AVG(pressure) as avg_pres,
            AVG(humidity) as avg_hum,
            MAX(wind_speed) as max_wind,
            AVG(sunshine) as avg_sun,
            MAX(precipitation) as max_perc
        FROM
            measurements
        WHERE
            mac_address = ? AND server_timestamp BETWEEN ? AND ?
        GROUP BY
            time_window
        ORDER BY
            time_window ASC;
    """
    
    # Przekazujemy do zapytania mac_address, dynamiczną datę początkową i końcową
    cur.execute(query, (mac_address, start_date, latest_timestamp))
    aggregated_data = [dict(row) for row in cur.fetchall()]
    return aggregated_data

# --- NOWY ENDPOINT API ---
@bp.route('/api/device_data/<mac_address>/aggregated', methods=['GET'])
def aggregated_device_data(mac_address):
    # Prosta weryfikacja, czy użytkownik ma dostęp (można rozbudować)
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Pobieramy dane z ostatnich 3 dni
        data = get_aggregated_data(mac_address, days=3)
        return jsonify(data)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania zagregowanych danych dla {mac_address}: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Zmieniona nazwa endpointu i URL
@bp.route('/device_data/<mac_address>', methods=['GET'])
def device_data_by_mac(mac_address):
    # 1. Sprawdzenie czy użytkownik jest zalogowany (pierwsza linia obrony)
    if 'username' not in session:
        logging.warning(f"Attempted access to /device_data/{mac_address} by unauthenticated user.")
        return redirect(url_for('login.login'))

    # 2. Pobranie nazwy użytkownika z sesji (bez zmian)
    username = session['username']
    
    # --- ZMIANA LOGIKI POBIERANIA DANYCH ---
    # Usuniemy starą logikę pobierania wszystkich pomiarów,
    # a w jej miejsce wstawimy wywołanie naszej nowej funkcji agregującej.

    try:
        # Weryfikacja, czy użytkownik ma dostęp do tego MAC adresu (ta logika pozostaje bez zmian)
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_row = cur.fetchone()
        if not user_row:
            abort(404, description="Użytkownik nie znaleziony.")
        user_id = user_row['id']
        cur.execute("SELECT COUNT(*) FROM user_boards WHERE user_id = ? AND mac_address = ?", (user_id, mac_address))
        
        if cur.fetchone()[0] == 0:
            abort(403, description="To urządzenie nie jest przypisane do Twojego konta.")

        cur.execute("""
            SELECT board_name FROM user_boards 
            WHERE user_id = ? AND mac_address = ?
        """, (user_id, mac_address))
        
        board_row = cur.fetchone()
        
        # 3. Obsłuż przypadki
        if board_row is None:
            # Jeśli nie znaleziono wiersza, użytkownik nie ma dostępu. Przerywamy.
            abort(403, description="To urządzenie nie jest przypisane do Twojego konta.")
        
        # Jeśli dotarliśmy tutaj, to znaczy, że board_row istnieje.
        # Teraz sprawdzamy, czy nazwa wewnątrz niego nie jest pusta (NULL).
        board_name = board_row['board_name']

        # NOWA CZĘŚĆ: Zamiast pobierać wszystkie pomiary, pobieramy zagregowane dane
        # Używamy funkcji zdefiniowanej w poprzednim kroku.
        # Przekazujemy wynik do szablonu pod nazwą 'measurements', aby uniknąć dużych zmian w HTML/JS.
        measurements = get_aggregated_data(mac_address, days=3)

        latest_conditions = None
        if measurements:
            # Bierzemy ostatni element z listy jako najnowsze warunki
            latest_conditions = measurements[-1]

    except sqlite3.Error as e:
        logging.error(f"Database error in device_data_by_mac: {e}")
        abort(500, description="Błąd bazy danych.")
    # --- KONIEC ZMIANY LOGIKI ---

    # 5. Renderowanie szablonu (bez zmian)
    return render_template(
        'device_data.html',
        mac_address=mac_address,
        board_name=board_name,
        username=username,
        measurements=measurements,
        latest_conditions=latest_conditions
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
            "SELECT server_timestamp, temperature, pressure, humidity, sunshine, wind_speed, precipitation FROM measurements WHERE mac_address = ? ORDER BY server_timestamp",
            (mac_address,)
        )
        all_measurements = cur.fetchall()

        if not all_measurements:
            # Brak danych pomiarowych DLA TEGO urządzenia, ale użytkownik je posiada
            return "Brak danych do pobrania dla tego urządzenia.", 404 # Zwróć 404 Not Found (danych)

        # 5. Konwersja danych na CSV i zwrócenie odpowiedzi (istniejąca logika)
        df = pd.DataFrame(all_measurements, columns=[
            'Timestamp', 'Temperatura (°C)', 'Ciśnienie (hPa)', 'Wilgotność (%)',
            'Wykryto Światło (Surowa Wartość)', 'Prędkość Wiatru (m/s)', 'Opady (%)'
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

@bp.route('/api/device_data/<mac_address>/available_dates', methods=['GET'])
def get_available_dates(mac_address):
    """Zwraca listę unikalnych dni (YYYY-MM-DD), dla których istnieją pomiary."""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        conn = get_db()
        cur = conn.cursor()
        # Zapytanie grupujące po dacie i zwracające unikalne dni
        cur.execute("""
            SELECT DISTINCT strftime('%Y-%m-%d', server_timestamp) as date
            FROM measurements
            WHERE mac_address = ?
            ORDER BY date DESC
        """, (mac_address,))
        
        # Zwracamy listę stringów z datami
        dates = [row['date'] for row in cur.fetchall()]
        return jsonify(dates)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania dostępnych dat dla {mac_address}: {e}")
        return jsonify({'error': 'Internal server error'}), 500
