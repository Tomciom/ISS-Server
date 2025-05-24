from routes import home, journey_history, latest_journey, login, register, journey_details, boards, device_data
from flask import Flask, jsonify, request, g
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
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE (user_id, mac_address)
    );
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
            wind_direction TEXT,
            rain_intensity_percent REAL            
        )
    ''')
    conn.commit()
    conn.close()

# helper to associate boards with users
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
            return True # Indicate successful insertion

    except sqlite3.IntegrityError:
        # This block is a fallback for race conditions, though the SELECT check minimizes its necessity.
        # It catches the UNIQUE constraint violation.
        logging.error(f"IntegrityError when adding device {mac_address} for user {username}. Duplicate entry?")
        # Depending on desired behavior, you might want to handle this differently.
        # Returning False here indicates it wasn't successfully inserted (likely due to duplicate).
        return False
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred in save_mac_to_db: {e}")
        # Re-raise the exception after logging it
        raise e

# save measurement record
def save_measurement(data):
    """
    Saves a measurement record to the measurements table.
    """
    try:
        # Use 'with' statement for automatic connection closing
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                '''
                INSERT INTO measurements (
                    mac_address,
                    server_timestamp, -- Use CURRENT_TIMESTAMP directly in SQL
                    temperature,
                    pressure,
                    humidity,
                    sunshine,
                    wind_speed,
                    wind_direction, 
                    rain_intensity_percent                  
                ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (\
                    data.get('mac_address'), # Use .get() for safer access
                    data.get('temperature'),
                    data.get('pressure'),
                    data.get('humidity'),
                    data.get('sunshine'),
                    data.get('wind_speed'),
                    data.get('wind_direction'),
                    data.get('rain_intensity_percent'),
                )
            )
            conn.commit() # Commit the transaction
    except Exception as e:
        logging.error(f"Error saving measurement: {e}")
        # Depending on requirements, you might want to re-raise or handle differently
        raise e

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
    app.register_blueprint(device_data.bp)

    @app.route('/<username>/add_device/<mac_address>', methods=['GET'])
    def add_device(username, mac_address):
        """
        API endpoint to associate a MAC address with a user.
        Handles cases for user not found, successful addition, and duplicate addition.
        """
        try:
            # Call the updated save_mac_to_db function
            success = save_mac_to_db(username, mac_address)
            
            if success:
                # Device was successfully added
                return jsonify({'message': f'Device {mac_address} added to user {username}'}), 200
            else:
                # save_mac_to_db returned False, indicating a duplicate
                # You might return 200 OK or 409 Conflict depending on your API design preference
                return jsonify({'message': f'Device {mac_address} is already associated with user {username}'}), 200 # Or 409

        except ValueError as e:
            # Handle the specific case where the user is not found
            logging.warning(f"Attempted to add device for non-existent user {username}: {e}")
            return jsonify({'error': str(e)}), 404 # Not Found

        except Exception as e:
            # Catch any other unexpected errors from save_mac_to_db
            logging.error(f"Error in add_device route: {e}")
            return jsonify({'error': 'An internal error occurred'}), 500

    @app.route('/<mac_address>/data', methods=['POST'])
    def receive_measurement(mac_address):
        """
        API endpoint to receive measurement data from a device.
        """
        data = request.get_json(force=True)
        logging.info(f"Received data for {mac_address}: {data}")

        # Ensure mac_address from URL matches data if provided, or add it
        # This adds robustness if the payload also contains mac_address
        if 'mac_address' in data and data['mac_address'] != mac_address:
             logging.warning(f"MAC address mismatch in URL ({mac_address}) and payload ({data['mac_address']})")
             # Decide how to handle this - maybe return an error?
             # For now, we'll use the one from the URL as it's part of the route
             pass # Or return jsonify({'error': 'MAC address mismatch'}), 400
        data['mac_address'] = mac_address # Ensure the mac_address used is from the URL

        # required fields check
        required = ['mac_address', 'temperature', 'pressure', 'humidity', 'sunshine', 'wind_speed', 'wind_direction', 'rain_intensity_percent']
        missing = [f for f in required if f not in data]
        if missing:
            msg = f"Missing fields in JSON: {', '.join(missing)}"
            logging.warning(msg)
            return jsonify({'error': msg}), 400

        try:
            save_measurement(data)
            # Use 201 Created status code for successful resource creation via POST
            return jsonify({'message': 'Measurement saved'}), 201
        except Exception as e:
            logging.error(f"Error in receive_measurement route: {e}")
            return jsonify({'error': 'Failed to save measurement'}), 500
        
        # Helper to get database connection
    def get_db():
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DB_PATH)
            db.row_factory = sqlite3.Row
        return db
    # Close database connection at the end of the request
    @app.teardown_appcontext
    def close_db(error):
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    return app




if __name__ == '__main__':
    init_db()
    app = create_app()
    # Consider using a more robust server like Gunicorn or uWSGI in production
    app.run(host="localhost", port=5000, debug=config.Config.DEBUG)
