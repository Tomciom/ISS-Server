from flask import Blueprint, request, jsonify
import threading
import uuid
import ai_main  # Importujemy nasz zrefaktoryzowany skrypt AI

bp = Blueprint('ai_service', __name__, url_prefix='/api/ai')

# Globalny "magazyn" na zadania i ich wyniki
# W produkcji lepiej użyć bazy danych lub Redis
tasks = {}

def run_ai_task(task_id, start_date, end_date):
    """Funkcja, która będzie uruchomiona w osobnym wątku."""
    print(f"Rozpoczynam zadanie AI: {task_id}")
    tasks[task_id]['status'] = 'RUNNING'
    try:
        # Wywołujemy główną funkcję z naszego modułu AI
        result = ai_main.run_prediction(start_date, end_date)
        tasks[task_id]['result'] = result
        tasks[task_id]['status'] = 'SUCCESS'
    except Exception as e:
        tasks[task_id]['status'] = 'FAILURE'
        tasks[task_id]['result'] = {'error': str(e)}
    print(f"Zakończono zadanie AI: {task_id} ze statusem {tasks[task_id]['status']}")

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