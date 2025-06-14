from flask import Blueprint, request, jsonify, g
import sqlite3
import threading
import uuid
import ai_main  # Importujemy nasz zrefaktoryzowany skrypt AI
from datetime import datetime, timedelta

bp = Blueprint('ai_service', __name__, url_prefix='/api/ai')

# Globalny "magazyn" na zadania i ich wyniki
# W produkcji lepiej użyć bazy danych lub Redis
tasks = {}

def run_ai_task(task_id, start_date, end_date):
    """Funkcja, która będzie uruchomiona w osobnym wątku."""
    print(f"Rozpoczynam zadanie AI: {task_id}")
    tasks[task_id]['status'] = 'RUNNING'
    try:
        result = ai_main.run_prediction(start_date, end_date)
        
        # --- NOWA LOGIKA OBSŁUGI WYNIKU ---
        # Sprawdzamy, czy wynik jest słownikiem z kluczem 'error'
        if isinstance(result, dict) and 'error' in result:
            # Jeśli tak, to jest to błąd zwrócony przez ai_main
            tasks[task_id]['status'] = 'FAILURE'
            tasks[task_id]['result'] = result # Przekazujemy słownik z błędem
        else:
            # W przeciwnym razie, wszystko jest ok
            tasks[task_id]['status'] = 'SUCCESS'
            tasks[task_id]['result'] = result
            
    except Exception as e:
        # Ten blok złapie błędy, które nie zostały obsłużone wewnątrz run_prediction
        import traceback
        print(f"KRYTYCZNY błąd w wątku AI dla zadania {task_id}: {e}")
        traceback.print_exc()
        tasks[task_id]['status'] = 'FAILURE'
        tasks[task_id]['result'] = {'error': 'Wystąpił nieoczekiwany błąd serwera.'}
        
    print(f"Zakończono zadanie AI: {task_id} ze statusem {tasks[task_id]['status']}")


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('measurements.db')
    return db

@bp.route('/check_data_availability', methods=['POST'])
def check_data_availability():
    data = request.get_json()
    if not data or 'target_timestamp' not in data:
        return jsonify({'error': 'Missing target_timestamp'}), 400

    try:
        target_dt = datetime.strptime(data['target_timestamp'], '%Y-%m-%d %H:%M:%S')
        start_check_dt = target_dt - timedelta(hours=48)
        
        conn = get_db()
        cur = conn.cursor()
        
        # --- ZMIANA 1: SPRAWDZENIE DANYCH DLA WYBRANEJ GODZINY ---
        # Sprawdzamy, czy istnieje jakikolwiek pomiar w godzinie, którą chcemy przewidzieć.
        # Tworzymy okno czasowe od HH:00:00 do HH:59:59.
        target_hour_start = target_dt.strftime('%Y-%m-%d %H:00:00')
        target_hour_end = target_dt.strftime('%Y-%m-%d %H:59:59')
        
        cur.execute("""
            SELECT 1 FROM measurements WHERE server_timestamp BETWEEN ? AND ? LIMIT 1
        """, (target_hour_start, target_hour_end))
        
        has_data_for_target_hour = cur.fetchone() is not None
        
        if not has_data_for_target_hour:
            return jsonify({'available': False, 'reason': 'Brak danych dla wybranej godziny.'})

        # --- ZMIANA 2: SPRAWDZENIE DANYCH HISTORYCZNYCH ---
        # Ta logika pozostaje, ale teraz wiemy, że godzina docelowa ma dane.
        cur.execute("""
            SELECT COUNT(DISTINCT strftime('%Y-%m-%d %H', server_timestamp))
            FROM measurements
            WHERE server_timestamp BETWEEN ? AND ?
        """, (start_check_dt.strftime('%Y-%m-%d %H:%M:%S'), target_dt.strftime('%Y-%m-%d %H:%M:%S')))
        
        hours_with_data = cur.fetchone()[0]
        
        is_available = hours_with_data > 38 # Wymagamy danych z ~80% okresu 48h
        
        reason = "Dane dostępne." if is_available else "Niewystarczająca ilość danych historycznych (48h)."
        
        return jsonify({'available': is_available, 'reason': reason})

    except Exception as e:
        print(f"Błąd w check_data_availability: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@bp.route('/predict', methods=['POST'])
def start_prediction():
    data = request.get_json()
    if not data or 'start_date' not in data or 'end_date' not in data:
        return jsonify({'error': 'Missing start_date or end_date'}), 400

    start_date = data['start_date']
    end_date = data['end_date']
    
    # Generujemy unikalne ID dla zadania
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'PENDING', 'result': None}

    # Uruchamiamy AI w osobnym wątku, aby nie blokować odpowiedzi
    thread = threading.Thread(target=run_ai_task, args=(task_id, start_date, end_date))
    thread.daemon = True # Wątek zakończy się, gdy główna aplikacja się zamknie
    thread.start()

    # Natychmiast zwracamy ID zadania
    return jsonify({'task_id': task_id}), 202 # 202 Accepted

@bp.route('/predict/status/<task_id>', methods=['GET'])
def get_prediction_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    response = {
        'task_id': task_id,
        'status': task['status']
    }
    # Jeśli zadanie się zakończyło, dołącz wynik
    if task['status'] in ['SUCCESS', 'FAILURE']:
        response['result'] = task['result']
    
    return jsonify(response)