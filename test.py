import sqlite3
import logging
import random # Dodajemy import modułu random
import time  # Dodajemy import modułu time

DB_PATH = 'measurements.db'

def generate_mock_measurement_data():
    """
    Generates a dictionary with mock measurement data.
    """
    mac_address = f"00:1A:2B:3C:4D:{random.randint(0, 255):02X}" # Prosty losowy MAC
    data = {
        'mac_address': mac_address,
        'temperature': round(random.uniform(-10.0, 40.0), 1),       # Stopnie Celsjusza
        'pressure': round(random.uniform(950.0, 1050.0), 1),      # hPa
        'humidity': round(random.uniform(0.0, 1.0), 2),         # Procenty
        'sunshine': round(random.uniform(0.0, 100.0), 1),         # Procenty (lub inna jednostka np. W/m^2)
        'wind_speed': round(random.uniform(0.0, 70.0), 1),        # km/h
        'precipitation': round(random.choices([0.0, random.uniform(0.1, 1.0)], weights=[0.80, 0.2])[0], 2) # mm, z większą szansą na 0
    }
    logging.debug(f"Generated mock data: {data}")
    return data

def save_mac_to_db(username, mac_address):
    """
    Associates a MAC address with a user in the user_boards table.
    Checks if the user exists and if the association already exists.
    Returns True on successful insertion, False if the association already exists.
    Raises ValueError if the user is not found.
    """
    try:
        # Use 'with' statement for automatic connection closing
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            
            # Find the user ID
            cur.execute("SELECT id FROM users WHERE username = ?", (username,))
            result = cur.fetchone()
            if not result:
                # User not found, raise a specific error
                raise ValueError(f"User '{username}' not found.")
            user_id = result[0]
            
            # Check if the user_id and mac_address pair already exists
            cur.execute(
                "SELECT 1 FROM user_boards WHERE user_id = ? AND mac_address = ?",
                (user_id, mac_address)
            )
            if cur.fetchone():
                # Pair already exists, log and return False
                logging.warning(f"Device {mac_address} already associated with user {username}")
                return False # Indicate that it was a duplicate

            # If not exists, insert the new record
            cur.execute(
                "INSERT INTO user_boards (user_id, mac_address) VALUES (?, ?)",
                (user_id, mac_address)
            )
            conn.commit() # Commit the transaction
            logging.info(f"Device {mac_address} successfully associated with user {username}")
            return True # Indicate successful insertion

    except sqlite3.IntegrityError:
        # This block is a fallback for race conditions, though the SELECT check minimizes its necessity.
        # It catches the UNIQUE constraint violation if the PK is (user_id, mac_address).
        logging.error(f"IntegrityError when adding device {mac_address} for user {username}. Duplicate entry?")
        return False
    except ValueError as ve: # Specifically catch ValueError for user not found
        logging.error(f"Error in save_mac_to_db: {ve}")
        raise ve # Re-raise to signal the issue
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred in save_mac_to_db: {e}")
        # Re-raise the exception after logging it
        raise e

def save_measurement(data=None, use_mock_data=False):
    """
    Saves a measurement record to the measurements table.
    The server_timestamp will be the current time + 2 hours.
    If 'use_mock_data' is True, random data will be generated and 'data' argument will be ignored.
    If 'use_mock_data' is False, 'data' argument must be provided.
    """
    if use_mock_data:
        data_to_save = generate_mock_measurement_data()
    elif data is not None:
        data_to_save = data
    else:
        logging.error("No data provided and use_mock_data is False. Cannot save measurement.")
        raise ValueError("Data must be provided if use_mock_data is False.")

    try:
        # Use 'with' statement for automatic connection closing
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                '''
                INSERT INTO measurements (
                    mac_address,
                    server_timestamp,
                    temperature,
                    pressure,
                    humidity,
                    sunshine,
                    wind_speed,
                    precipitation
                ) VALUES (?, datetime(CURRENT_TIMESTAMP, '+2 hours'), ?, ?, ?, ?, ?, ?)
                ''',
                (
                    data_to_save.get('mac_address'),
                    data_to_save.get('temperature'),
                    data_to_save.get('pressure'),
                    data_to_save.get('humidity'),
                    data_to_save.get('sunshine'),
                    data_to_save.get('wind_speed'),
                    data_to_save.get('precipitation'),
                )
            )
            conn.commit() # Commit the transaction
            log_msg = f"Measurement saved for MAC: {data_to_save.get('mac_address')}"
            if use_mock_data:
                log_msg += " (mocked data)"
            logging.info(log_msg)
    except sqlite3.Error as e:
        sql_statement = "N/A"
        # Check if 'cur' was defined and has 'statement' (might not be if connection failed)
        if 'cur' in locals() and hasattr(cur, 'statement'):
            sql_statement = cur.statement if cur.statement else "N/A"
        logging.error(f"SQLite error saving measurement: {e} - SQL: {sql_statement}")
        raise e
    except Exception as e:
        logging.error(f"General error saving measurement: {e}")
        raise e

# Przykładowe użycie (do testów):
if __name__ == '__main__':
    DB_PATH = 'measurements.db' # Użyj testowej bazy danych
    # Ustawienie poziomu logowania na DEBUG, aby widzieć generowane mock dane
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Utwórz tabele, jeśli nie istnieją (do celów testowych)
    try:
        with sqlite3.connect(DB_PATH) as conn_setup:
            # Tabela użytkowników
            conn_setup.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL
                )
            ''')
            # Tabela powiązań użytkownik-MAC
            conn_setup.execute('''
                CREATE TABLE IF NOT EXISTS user_boards (
                    user_id INTEGER,
                    mac_address TEXT,
                    PRIMARY KEY (user_id, mac_address),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            # Tabela pomiarów
            conn_setup.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mac_address TEXT,
                    server_timestamp TIMESTAMP,
                    temperature REAL,
                    pressure REAL,
                    humidity REAL,
                    sunshine REAL,
                    wind_speed REAL,
                    precipitation REAL
                )
            ''')
            conn_setup.commit()
            logging.info("Tabele 'users', 'user_boards', 'measurements' gotowe.")

            # Dodaj użytkownika "admin", jeśli nie istnieje
            conn_setup.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", ('admin',))
            conn_setup.commit()
            logging.info("Użytkownik 'admin' gotowy.")

    except Exception as e_setup:
        logging.error(f"Nie udało się przygotować bazy danych: {e_setup}")
        exit()

    # Przypisanie MAC adresu do użytkownika "admin"
    admin_user = "admin"
    mac_to_assign = 'AA:BB:CC:DD:EE:FF' # Ten sam MAC co w sample_data poniżej

    logging.info(f"Próba przypisania MAC {mac_to_assign} do użytkownika '{admin_user}'...")
    try:
        if save_mac_to_db(admin_user, mac_to_assign):
            logging.info(f"Pomyślnie przypisano MAC {mac_to_assign} do '{admin_user}'.")
        else:
            logging.info(f"MAC {mac_to_assign} był już wcześniej przypisany do '{admin_user}' lub wystąpił inny problem z zapisem.")
    except ValueError as ve:
        logging.error(f"Nie udało się przypisać MAC: {ve}")
    except Exception as e_assign:
        logging.error(f"Nieoczekiwany błąd podczas przypisywania MAC: {e_assign}")
    
    # Ponowna próba przypisania tego samego MAC, aby sprawdzić logikę duplikatu
    logging.info(f"Próba ponownego przypisania MAC {mac_to_assign} do użytkownika '{admin_user}' (test duplikatu)...")
    try:
        if save_mac_to_db(admin_user, mac_to_assign):
            # To nie powinno się zdarzyć, jeśli pierwsze przypisanie było udane
            logging.warning(f"MAC {mac_to_assign} został przypisany, mimo że powinien być duplikatem.")
        else:
            logging.info(f"Ponowne przypisanie MAC {mac_to_assign} do '{admin_user}' nie powiodło się (zgodnie z oczekiwaniami dla duplikatu).")
    except Exception as e_assign_dup:
        logging.error(f"Nieoczekiwany błąd podczas ponownego przypisywania MAC: {e_assign_dup}")


    # Przykład 1: Zapis konkretnych danych (z użyciem przypisanego MAC)
    sample_data = {
        'mac_address': mac_to_assign, # Używamy MAC przypisanego do admina
        'temperature': 22.3,
        'pressure': 1010.1,
        'humidity': 0.55,
        'sunshine': 80.0,
        'wind_speed': 15.2,
        'precipitation': 0.5
    }
    try:
        save_measurement(data=sample_data)
    except Exception as e_main:
        logging.error(f"Wystąpił błąd podczas zapisywania (konkretne dane): {e_main}")


    time.sleep(1) 
    # Przykład 2: Zapis mockowanych danych (będą miały losowe MAC adresy)
    try:
        save_measurement(use_mock_data=True)
    except Exception as e_main:
        logging.error(f"Wystąpił błąd podczas zapisywania (mockowane dane): {e_main}")
        
    # Przykład 3: Zapis kilku mockowanych rekordów w pętli
    logging.info("Zapisywanie 3 dodatkowych mockowanych pomiarów...")
    for i in range(3):
        try:
            time.sleep(1) # Dodajemy opóźnienie między zapisami
            save_measurement(use_mock_data=True)
        except Exception as e_loop:
            logging.error(f"Błąd przy zapisie mockowanego pomiaru {i+1}/3: {e_loop}")


    # Weryfikacja (opcjonalnie) - odczyt kilku ostatnich rekordów z measurements
    try:
        with sqlite3.connect(DB_PATH) as conn_verify:
            cur_verify = conn_verify.cursor()
            logging.info("Ostatnie 5 zapisów w tabeli 'measurements':")
            cur_verify.execute("SELECT id, mac_address, server_timestamp, temperature FROM measurements ORDER BY id DESC LIMIT 5")
            rows = cur_verify.fetchall()
            if rows:
                for row in rows:
                    logging.info(f"Odczytano z 'measurements': ID={row[0]}, MAC={row[1]}, Timestamp={row[2]}, Temp={row[3]}")
            else:
                logging.warning("Nie znaleziono zapisów w 'measurements' do weryfikacji.")

            # Weryfikacja zawartości tabeli 'user_boards'
            logging.info("Zawartość tabeli 'user_boards':")
            cur_verify.execute("SELECT ub.mac_address, u.username FROM user_boards ub JOIN users u ON ub.user_id = u.id")
            user_board_rows = cur_verify.fetchall()
            if user_board_rows:
                for row in user_board_rows:
                    logging.info(f"Odczytano z 'user_boards': MAC={row[0]} przypisany do User={row[1]}")
            else:
                logging.warning("Tabela 'user_boards' jest pusta.")

    except Exception as e_verify:
        logging.error(f"Błąd podczas weryfikacji danych z bazy: {e_verify}")