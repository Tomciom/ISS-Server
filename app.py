from flask import Flask, jsonify, request
from routes import home, journey_history,latest_journey, login, register, new_journey, journey_details, boards
import config, sqlite3
import sqlite3
from flask_cors import CORS


def create_journeys():
    conn = sqlite3.connect('journeys.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS journeys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            is_current BOOLEAN DEFAULT 0,
            mac_address TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS temperature_pressure (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            journey_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            temperature REAL NOT NULL,
            pressure REAL NOT NULL,
            mac_address TEXT,
            FOREIGN KEY (journey_id) REFERENCES journeys (id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS fire_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            journey_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            fire_detected INTEGER NOT NULL,
            sensor_value REAL NOT NULL,
            mac_address TEXT,
            FOREIGN KEY (journey_id) REFERENCES journeys (id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS rotation_acceleration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            journey_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            rotation_degrees_x REAL NOT NULL,
            rotation_degrees_y REAL NOT NULL,
            rotation_degrees_z REAL NOT NULL,
            gx REAL NOT NULL,
            gy REAL NOT NULL,
            gz REAL NOT NULL,
            mac_address TEXT,
            FOREIGN KEY (journey_id) REFERENCES journeys (id)
        )
    ''')

    conn.commit()
    conn.close()


def init_db():



    conn = sqlite3.connect('Users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_boards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    mac_address TEXT NOT NULL,
                    board_name TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    
    conn.commit()


    conn.close()

def save_mac_to_db(username, mac_address):
    conn_users = sqlite3.connect('Users.db')
    cur_users = conn_users.cursor()

    cur_users.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cur_users.fetchone()
    if result is None:
        conn_users.close()
        raise ValueError(f"User with username '{username}' not found.")
    user_id = result[0]

    cur_users.execute(
        "INSERT INTO user_boards (user_id, mac_address) VALUES (?, ?)",
        (user_id, mac_address)
    )
    conn_users.commit()
    conn_users.close()

    remove_journeys = False

    if remove_journeys:
        conn_journeys = sqlite3.connect('journeys.db')
        cur_journeys = conn_journeys.cursor()

        cur_journeys.execute(
            "SELECT id FROM journeys WHERE mac_address = ?",
            (mac_address,)
        )
        journey_ids = cur_journeys.fetchall()

        for (journey_id,) in journey_ids:
            cur_journeys.execute(
                "DELETE FROM temperature_pressure WHERE journey_id = ?",
                (journey_id,)
            )
            cur_journeys.execute(
                "DELETE FROM fire_detection WHERE journey_id = ?",
                (journey_id,)
            )
            cur_journeys.execute(
                "DELETE FROM rotation_acceleration WHERE journey_id = ?",
                (journey_id,)
            )
        
        cur_journeys.execute(
            "DELETE FROM journeys WHERE mac_address = ?",
            (mac_address,)
        )
        conn_journeys.commit()
        conn_journeys.close()



def send_mac():
    try:
        received_data = request.json
        print("Received data:", received_data)
        save_mac_to_db(received_data['username'], received_data['mac_address'])
        response = {"response": f"Data received: {received_data['mac_address']}"}
        return jsonify(response), 200
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Invalid request"}), 400

    
def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    CORS(app)

    app.register_blueprint(boards.bp)
    app.register_blueprint(home.bp)
    app.register_blueprint(journey_history.bp)
    app.register_blueprint(journey_details.bp)
    app.register_blueprint(latest_journey.bp)
    app.register_blueprint(login.bp)
    app.register_blueprint(register.bp)
    app.register_blueprint(new_journey.bp)

    app.add_url_rule('/send_mac', view_func=send_mac, methods=['POST'])
    
    @app.route('/<username>/add_device/<mac_address>', methods=['GET'])
    def add_device(username, mac_address):
        try:
            save_mac_to_db(username, mac_address)
            return jsonify({
                'message': f'Device {mac_address} successfully added to user {username}'
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return app

if __name__ == '__main__':
    init_db()
    create_journeys()
    app = create_app()
    app.run(host=config.mqtt_broker, port=5000,debug=config.Config.DEBUG)