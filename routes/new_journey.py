from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import paho.mqtt.client as mqtt
import json
import config
from time import sleep as Sleep

bp = Blueprint('new_journey', __name__)

mqtt_broker = config.mqtt_broker
mqtt_port = 1883

@bp.route('/new_journey', methods=['GET', 'POST'])
def new_journey():
    username = session.get('username')
    if request.method == 'GET':
        username = session.get('username')
        if not username:
            return redirect(url_for('login.login'))

        conn = sqlite3.connect('Users.db')
        c = conn.cursor()

        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id_row = c.fetchone()

        boards = []
        if user_id_row:
            user_id = user_id_row[0]
            c.execute("SELECT mac_address, board_name FROM user_boards WHERE user_id = ?", (user_id,))
            boards = [{'mac_address': row[0], 'board_name': row[1]} for row in c.fetchall()]

        conn.close()
        return render_template('new_journey.html', username=username, boards=boards)

    if request.method == 'POST':
        username = session.get('username')
        data = request.json
        try:
            print("Received configuration:", data)
            mac = data['mac_address']
            data.pop('journey_name', None)
            data.pop('mac_address', None)
            print(f"json: {data}")

            mqtt_client = mqtt.Client()
            mqtt_client.connect(mqtt_broker, mqtt_port, 60)

            mqtt_topic = f"BlackBox/{username}/{mac}/Config"
            print(f"Publishing to topic {mqtt_topic}: {json.dumps(data)}")
            mqtt_client.publish(mqtt_topic, json.dumps(data))

            messages_received = []

            def on_message(client, userdata, msg):
                messages_received.append(msg)

            mqtt_client.on_message = on_message
            topic = "BlackBox/+/+/ConfigResponse"
            mqtt_client.subscribe(topic)

            mqtt_client.loop_start()
            Sleep(1)
            mqtt_client.loop_stop()

            for msg in messages_received:
                try:
                    topic_parts = msg.topic.split("/")
                    topic_username = topic_parts[1]
                    topic_mac = topic_parts[2]
                    pin_received = msg.payload.decode("utf-8")
                    print(f"Otrzymano wiadomość dla użytkownika: {topic_username}, MAC: {topic_mac}, PIN: {pin_received}")

                    if topic_username == username:
                        conn = sqlite3.connect('Users.db')
                        c = conn.cursor()
                        c.execute("SELECT id FROM users WHERE username = ?", (username,))
                        user_id = c.fetchone()
                        if user_id:
                            user_id = user_id[0]
                            c.execute("UPDATE user_boards SET is_in_use = ? WHERE user_id = ?", (int(pin_received), int(user_id)))
                            conn.commit()
                            print(f"Aktualizacja PIN {pin_received} dla użytkownika {username} zakończona pomyślnie.")
                        conn.close()
                except Exception as e:
                    print(f"Błąd podczas przetwarzania wiadomości: {e}")

            return redirect(url_for('home.home'))

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'message': 'Wystąpił błąd podczas wysyłania danych przez MQTT.'}), 500

    return render_template('new_journey.html',username=username, boards=boards)

