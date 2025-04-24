from routes import home, journey_history, latest_journey, login, register, journey_details, boards
from flask import Flask, jsonify, request
import config
import sqlite3
import logging

DB_PATH = 'measurements.db'

# initialize the measurements database and tables
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # users and user_boards tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_boards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mac_address TEXT NOT NULL,
            board_name TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    # measurements table: include client timestamp, sunshine, wind data
    c.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mac_address TEXT NOT NULL,
            server_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature REAL,
            pressure REAL,
            humidity REAL,
            sunshine INTEGER,
            wind_speed REAL,
            wind_direction TEXT
        )
    ''')
    conn.commit()
    conn.close()

# helper to associate boards with users
def save_mac_to_db(username, mac_address):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cur.fetchone()
    if not result:
        conn.close()
        raise ValueError(f"User '{username}' not found.")
    user_id = result[0]
    cur.execute(
        "INSERT INTO user_boards (user_id, mac_address) VALUES (?, ?)",
        (user_id, mac_address)
    )
    conn.commit()
    conn.close()

# save measurement record
def save_measurement(data):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO measurements (
            mac_address,
            temperature,
            pressure,
            humidity,
            sunshine,
            wind_speed,
            wind_direction
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            data['mac_address'],
            data['temperature'],
            data['pressure'],
            data['humidity'],
            data['sunshine'],
            data['wind_speed'],
            data['wind_direction'],
        )
    )
    conn.commit()
    conn.close()

# create Flask app and routes
def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # enable request logging
    logging.basicConfig(level=logging.INFO)

    # register blueprints
    app.register_blueprint(boards.bp)
    app.register_blueprint(home.bp)
    app.register_blueprint(journey_history.bp)
    app.register_blueprint(journey_details.bp)
    app.register_blueprint(latest_journey.bp)
    app.register_blueprint(login.bp)
    app.register_blueprint(register.bp)

    @app.route('/<username>/add_device/<mac_address>', methods=['GET'])
    def add_device(username, mac_address):
        try:
            save_mac_to_db(username, mac_address)
            return jsonify({'message': f'Device {mac_address} added to user {username}'})
        except Exception as e:
            logging.error(f"Error add_device: {e}")
            return jsonify({'error': str(e)}), 400

    @app.route('/<mac_address>/data', methods=['POST'])
    def receive_measurement(mac_address):
        data = request.get_json(force=True)
        logging.info(f"Received data for {mac_address}: {data}")
        data['mac_address'] = mac_address

        # required fields
        required = ['mac_address', 'temperature', 'pressure', 'humidity', 'sunshine', 'wind_speed', 'wind_direction']
        missing = [f for f in required if f not in data]
        if missing:
            msg = f"Missing fields in JSON: {', '.join(missing)}"
            logging.warning(msg)
            return jsonify({'error': msg}), 400

        try:
            save_measurement(data)
            return jsonify({'message': 'Measurement saved'})
        except Exception as e:
            logging.error(f"Error save_measurement: {e}")
            return jsonify({'error': str(e)}), 500

    return app

if __name__ == '__main__':
    init_db()
    app = create_app()
    app.run(host="192.168.1.15", port=5000, debug=config.Config.DEBUG)
