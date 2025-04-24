from flask import Blueprint, render_template, session, Response
import sqlite3
import csv, io

bp = Blueprint('journey_details', __name__)

def get_journey_data(journey_id):
    conn = sqlite3.connect('journeys.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT timestamp, temperature, pressure FROM temperature_pressure WHERE journey_id = ?', (journey_id,))
    measurements = cursor.fetchall()

    cursor.execute('SELECT timestamp, fire_detected, sensor_value FROM fire_detection WHERE journey_id = ?', (journey_id,))
    fire_data = cursor.fetchall()

    cursor.execute('SELECT timestamp, rotation_degrees_x, rotation_degrees_y, rotation_degrees_z, gx, gy, gz FROM rotation_acceleration WHERE journey_id = ?', (journey_id,))
    rotation_data = cursor.fetchall()

    conn.close()
    return measurements, fire_data, rotation_data

@bp.route('/journey_details/<int:journey_id>', methods=['GET'])
def journey_details(journey_id):
    username = session.get('username')
    measurements, fire_data, rotation_data = get_journey_data(journey_id)
    return render_template('journey_details.html', 
                           journey_id=journey_id, 
                           measurements=measurements, 
                           fire_data=fire_data, 
                           rotation_data=rotation_data,
                           username=username)


@bp.route('/journey_details/<int:journey_id>/download_csv')
def download_csv(journey_id):
    measurements, fire_data, rotation_data = get_journey_data(journey_id)
    
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['Pomiary'])
    writer.writerow(['Timestamp', 'Temperatura (°C)', 'Ciśnienie (hPa)'])
    for row in measurements:
        writer.writerow([row['timestamp'], row['temperature'], row['pressure']])
    writer.writerow([])

    writer.writerow(['Wykrycie światła'])
    writer.writerow(['Timestamp', 'Wykryto światło', 'Wartość czujnika'])
    for row in fire_data:
        writer.writerow([row['timestamp'], row['fire_detected'], row['sensor_value']])
    writer.writerow([])

    writer.writerow(['Prędkość kątowa i Przyspieszenie'])
    writer.writerow(['Timestamp', 'Prędkość w x (°/s)', 'Prędkość w y (°/s)', 'Prędkość w z (°/s)', 'GX', 'GY', 'GZ'])
    for row in rotation_data:
        writer.writerow([
            row['timestamp'], 
            row['rotation_degrees_x'], 
            row['rotation_degrees_y'], 
            row['rotation_degrees_z'], 
            row['gx'], 
            row['gy'], 
            row['gz']
        ])

    output.seek(0)
    csv_output = output.getvalue()
    output.close()

    return Response(
        csv_output,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=journey_{journey_id}.csv"}
    )
