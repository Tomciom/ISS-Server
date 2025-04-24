import paho.mqtt.client as mqtt
import routes.home as home
import routes.new_journey as new_journey
import requests, json, config

BROKER_ADDRESS = config.mqtt_broker
BROKER_PORT = 1883
TOPIC_RECEIVE = "BlackBox/+/+/Pin"
TOPIC_RESPONSE_TEMPLATE = "BlackBox/{username}/{mac}/Response"

def send_config(json, username):
    mac = json['mac_address']
    json.pop('journey_name')
    json.pop('mac_address')
    topic = "BlackBox/{username}/{mac}/Response"
    client.publish(topic, json)


def fetch_pin(username):
    print(f"Fetching PIN for user: {username}")
    pin = home.send_code(username)
    print(f"Received data: {pin}")
    return pin

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(TOPIC_RECEIVE)

def on_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split("/")
        username = topic_parts[1]
        mac = topic_parts[2]
        pin_received = msg.payload.decode("utf-8")

        print(f"Received message:")
        print(f"Username (from topic): {username}")
        print(f"MAC: {mac}")
        print(f"PIN (from payload): {pin_received}")

        code_data = fetch_pin(username)

        print(f"Received data: {code_data}")

        if code_data == pin_received:
            response = {"mac_address": mac, "username": username, "value": 100}
            response_esp = "1"
        
            post_response = requests.post(('http://' + BROKER_ADDRESS + ':5000/send_mac'), json=response)
            print(f"Sent POST response: {post_response.status_code}")
        else:
            print("Username or PIN did not match. No POST request sent.")
            response_esp = "0"
        
        response_topic = TOPIC_RESPONSE_TEMPLATE.format(username=username, mac=mac)
        client.publish(response_topic, response_esp)

    except Exception as e:
        print(f"Error processing message: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

client.loop_forever()
