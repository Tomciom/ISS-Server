# -*- coding: utf-8 -*-
from datetime import datetime, timedelta # ZMIENIONE: Dodano timedelta
import pandas as pd
import numpy as np
import math
import time
import warnings
import os
import joblib
import sqlite3 # NOWE: Do obsługi bazy danych

# Ignoruj ostrzeżenia (bez zmian)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- === Konfiguracja Skryptu === ---
# Ustawienia oryginalne (mogą być mniej istotne dla trybu predykcji z bazy)
USE_SMOTE = True
LOAD_MODELS_IF_EXIST = True # MUSI być True dla predykcji
FORCE_RETRAIN = False       # MUSI być False dla predykcji
MODEL_SAVE_DIR = "trained_models_wien"
# --- ========================== ---

# --- === NOWA KONFIGURACJA DLA DANYCH Z BAZY === ---
USE_DATABASE_INPUT = True # Ustaw na True, aby używać danych z bazy
DB_PATH = 'measurements.db' # Ścieżka do Twojej bazy SQLite

# Parametry dla przetwarzania danych z bazy (DOSTOSUJ DO SWOICH CZUJNIKÓW!)
SUNSHINE_THRESHOLD = 30
MAX_PRECIP_RATE_MM_PER_HOUR_FOR_INTENSITY_1 = 25.0 # mm/h dla prcp_intensity = 1.0
TEMP_THRESHOLD_SNOW = 0.5  # Temperatura (C) poniżej której opad jest śniegiem
WATER_TO_SNOW_RATIO = 10.0 # mm śniegu z 1 mm wody
HOURLY_MELT_RATE_PER_DEG_ABOVE_FREEZING = 1.0 # mm ekwiwalentu wodnego topnienia / godzinę / °C > 0
MAX_ACCUMULATED_SNOW_DEPTH_MM = 1000 # Max symulowana grubość pokrywy śnieżnej w mm

# Zakres czasu dla predykcji (PRZYKŁADOWE WARTOŚCI - DOSTOSUJ!)
# Aby uniknąć błędów, upewnij się, że masz dane w bazie dla tego zakresu.
PREDICTION_START_DATE = datetime(2025, 5, 25, 14, 0, 0) # Początek okresu predykcji
PREDICTION_END_DATE = datetime(2025, 5, 25, 14, 59, 59)   # Koniec okresu predykcji

# Dane z bazy potrzebne do obliczenia cech opóźnionych/kroczących
# Pobierz dane np. 48h wcześniej niż początek predykcji
DB_DATA_FETCH_BUFFER_HOURS = 48 # Ile godzin danych historycznych przed PREDICTION_START_DATE
DB_DATA_FETCH_START_DATE = PREDICTION_START_DATE - timedelta(hours=DB_DATA_FETCH_BUFFER_HOURS)
DB_DATA_FETCH_END_DATE = PREDICTION_END_DATE # Pobieramy dane aż do końca okresu predykcji
# --- ======================================= ---

# 1. Import bibliotek
try:
    from meteostat import Hourly, Stations # Meteostat będzie używany tylko jeśli USE_DATABASE_INPUT=False
except ImportError:
    print("OSTRZEŻENIE: Biblioteka Meteostat nie jest zainstalowana. Wymagane, jeśli USE_DATABASE_INPUT=False.")
    if not USE_DATABASE_INPUT:
        print("BŁĄD: Meteostat wymagany i niedostępny. Przerywam.")
        exit()

try:
    from sklearn.model_selection import train_test_split # Mniej istotne dla predykcji
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import class_weight # Mniej istotne dla predykcji
    SKLEARN_AVAILABLE = True
except ImportError:
    print("BŁĄD: Kluczowe biblioteki scikit-learn nie są zainstalowane.")
    exit()
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("BŁĄD: Biblioteka 'xgboost' nie jest zainstalowana.")
    XGB_AVAILABLE = False
    exit()
try:
    from imblearn.over_sampling import SMOTE # Mniej istotne dla predykcji
    SMOTE_AVAILABLE = True
except ImportError:
    print("OSTRZEŻENIE: Biblioteka 'imblearn' (dla SMOTE) nie jest zainstalowana.")
    SMOTE_AVAILABLE = False
    if USE_SMOTE: print("   Wyłączam USE_SMOTE."); USE_SMOTE = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("Ostrzeżenie: Wizualizacja (matplotlib, seaborn) niedostępna.")

# Utwórz katalog na modele (bez zmian)
if not os.path.exists(MODEL_SAVE_DIR):
    try:
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Utworzono katalog na modele: {MODEL_SAVE_DIR}")
    except OSError as e:
        print(f"BŁĄD: Nie można utworzyć katalogu na modele '{MODEL_SAVE_DIR}': {e}")
        MODEL_SAVE_DIR = "."
        print(f"Modele będą zapisywane w katalogu bieżącym.")


# 2. Funkcja do obliczania punktu rosy (bez zmian)
def calculateDewPoint(temperature, humidity):
    if pd.isna(temperature) or pd.isna(humidity) or humidity <= 0 or humidity > 100: return np.nan
    try: a=17.27; b=237.7; gamma=(a * temperature) / (b + temperature) + math.log(humidity / 100.0); dewPoint = (b * gamma) / (a - gamma)
    except (ValueError, OverflowError): return np.nan
    if dewPoint > temperature + 0.5 : return temperature
    if dewPoint < -80: return np.nan
    return dewPoint

# 3. Funkcja Agregacji Coco (bez zmian)
def aggregate_coco_FINAL_user_v2(coco_code):
    if pd.isna(coco_code): return "Unknown"
    try: code = int(coco_code)
    except (ValueError, TypeError): return "Unknown"
    if code in [1, 2]: return "Clear/Fair"
    if code in [3, 4]: return "Cloudy/Overcast"
    if code in [5, 6]: return "Fog"
    if code in [7, 8, 9, 17, 18]: return "Rain"
    if code in [10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22]: return "Snow/Sleet/Freezing"
    if code in [23, 24, 25, 26, 27]: return "Thunderstorm/Severe"
    return "Unknown"

# Definicje kategorii pogodowych (potrzebne wcześniej dla bloku inżynierii cech)
no_precip_categories_user = ['Clear/Fair', 'Cloudy/Overcast', 'Fog']
precip_categories_user = ['Rain', 'Snow/Sleet/Freezing', 'Thunderstorm/Severe']
all_categories_user = sorted(no_precip_categories_user + precip_categories_user)


# --- Ustawienia Danych (Meteostat - mniej istotne jeśli USE_DATABASE_INPUT=True) ---
target_station_ids = ['11036']
station_names = {'11036': 'Wien / Schwechat'}
train_start_year = 2018; train_end_year = 2023; test_year = 2024 # Dla trybu Meteostat
data_fetch_start_date_meteostat = datetime(train_start_year, 1, 1)
data_fetch_end_date_meteostat = datetime(test_year, 12, 31, 23, 59, 59)

print(f"--- Hierarchiczny Model XGBoost v6 ---")
if USE_DATABASE_INPUT:
    print(f"   TRYB: Predykcja na danych z bazy SQLite")
    print(f"   Ścieżka do bazy: {DB_PATH}")
    print(f"   Okres pobierania danych z bazy: {DB_DATA_FETCH_START_DATE.strftime('%Y-%m-%d %H:%M')} - {DB_DATA_FETCH_END_DATE.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Okres predykcji: {PREDICTION_START_DATE.strftime('%Y-%m-%d %H:%M')} - {PREDICTION_END_DATE.strftime('%Y-%m-%d %H:%M')}")
else:
    print(f"   TRYB: Trening/Test na danych Meteostat dla stacji: {', '.join([f'{station_names.get(sid, sid)} ({sid})' for sid in target_station_ids])}")
    print(f"   Okres pobierania danych Meteostat: {data_fetch_start_date_meteostat.strftime('%Y-%m-%d')} - {data_fetch_end_date_meteostat.strftime('%Y-%m-%d')}")
    print(f"   Zbiór treningowy: Lata {train_start_year}-{train_end_year}")
    print(f"   Zbiór testowy: Rok {test_year}")
print(f"   (Wczytywanie modeli: {'Tak' if LOAD_MODELS_IF_EXIST else 'Nie'}, Wymuszony trening: {'Tak' if FORCE_RETRAIN else 'Nie'})")


# --- Pobieranie i Przetwarzanie Danych Wejściowych ---
print("\n--- Etap 1: Przygotowanie Danych Wejściowych ---")
full_processing_start_time = time.time()
df_for_feature_engineering = None # DataFrame, który trafi do inżynierii cech

if USE_DATABASE_INPUT:
    print("  Pobieranie i przetwarzanie danych z bazy SQLite...")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"""
            SELECT server_timestamp, temperature, pressure, humidity, sunshine, wind_speed, precipitation
            FROM measurements
            WHERE server_timestamp BETWEEN ? AND ?
            ORDER BY server_timestamp ASC
        """
        # Użycie parametrów w zapytaniu SQL dla bezpieczeństwa
        df_raw = pd.read_sql_query(query, conn, params=(DB_DATA_FETCH_START_DATE.strftime('%Y-%m-%d %H:%M:%S'),
                                                        DB_DATA_FETCH_END_DATE.strftime('%Y-%m-%d %H:%M:%S')))
        conn.close()

        if df_raw.empty:
            print(f"  BŁĄD: Brak danych w bazie dla zadanego okresu: {DB_DATA_FETCH_START_DATE} - {DB_DATA_FETCH_END_DATE}. Przerywam.")
            exit()

        df_raw['server_timestamp'] = pd.to_datetime(df_raw['server_timestamp'])
        df_raw.set_index('server_timestamp', inplace=True)
        print(f"    Wczytano {len(df_raw)} surowych rekordów z bazy.")

        df_converted = df_raw.copy()
        df_converted.rename(columns={
            'temperature': 'temp_celsius', 'pressure': 'pres_hpa', 'humidity': 'rhum_fraction',
            'sunshine': 'sunshine_analog', 'wind_speed': 'wspd_kmh', 'precipitation': 'prcp_intensity'
        }, inplace=True)

        df_converted['temp'] = df_converted['temp_celsius']
        df_converted['pres'] = df_converted['pres_hpa']
        df_converted['rhum'] = df_converted['rhum_fraction'] * 100
        df_converted['wspd_mps'] = df_converted['wspd_kmh'] / 3.6
        df_converted['is_sunny_interval'] = (df_converted['sunshine_analog'] > SUNSHINE_THRESHOLD).astype(int)
        df_converted['sunshine_minutes_interval'] = df_converted['is_sunny_interval'] * (5.0 / 60.0)
        _MAX_PRECIP_MM_PER_5SEC_FOR_INTENSITY_1 = (MAX_PRECIP_RATE_MM_PER_HOUR_FOR_INTENSITY_1 / 3600.0) * 5.0
        df_converted['prcp_mm_interval'] = df_converted['prcp_intensity'] * _MAX_PRECIP_MM_PER_5SEC_FOR_INTENSITY_1
        
        final_cols_before_agg = ['temp', 'pres', 'rhum', 'wspd_mps', 'sunshine_minutes_interval', 'prcp_mm_interval']
        current_cols = df_converted.columns.tolist()
        missing_raw_cols = [col for col in final_cols_before_agg if col not in current_cols]
        if missing_raw_cols:
            print(f"    BŁĄD: Brakuje kolumn do agregacji po konwersji: {missing_raw_cols}. Sprawdź nazwy w bazie i logikę konwersji.")
            exit()
        df_for_aggregation = df_converted[final_cols_before_agg].copy()
        print("    Dokonano konwersji jednostek z danych bazodanowych.")

        agg_functions_db = {
            'temp': 'mean', 'pres': 'mean', 'rhum': 'mean',
            'wspd_mps': ['mean', 'max'],
            'sunshine_minutes_interval': 'sum', 'prcp_mm_interval': 'sum'
        }
        _cols_to_fill_na_db = ['wspd_mps', 'sunshine_minutes_interval', 'prcp_mm_interval', 'temp', 'pres', 'rhum']
        for col in _cols_to_fill_na_db:
            if col in df_for_aggregation.columns: df_for_aggregation[col].fillna(0, inplace=True) # Wypełnij NaN przed agregacją

        if df_for_aggregation.empty: print("    Brak danych do agregacji."); exit()
        df_hourly_multiindex = df_for_aggregation.resample('H').agg(agg_functions_db)
        if df_hourly_multiindex.empty: print("    Agregacja nie dała wyników (pusty DataFrame)."); exit()
        
        df_hourly_from_db = pd.DataFrame()
        df_hourly_from_db['temp'] = df_hourly_multiindex[('temp', 'mean')]
        df_hourly_from_db['pres'] = df_hourly_multiindex[('pres', 'mean')]
        df_hourly_from_db['rhum'] = df_hourly_multiindex[('rhum', 'mean')]
        df_hourly_from_db['wspd'] = df_hourly_multiindex[('wspd_mps', 'mean')]
        df_hourly_from_db['wpgt'] = df_hourly_multiindex[('wspd_mps', 'max')]
        df_hourly_from_db['tsun'] = df_hourly_multiindex[('sunshine_minutes_interval', 'sum')]
        df_hourly_from_db['prcp'] = df_hourly_multiindex[('prcp_mm_interval', 'sum')]
        
        df_hourly_from_db['tsun'] = np.clip(df_hourly_from_db['tsun'], 0, 60)
        df_hourly_from_db['rhum'] = np.clip(df_hourly_from_db['rhum'], 0, 100)
        print(f"    Zagregowano dane z bazy do {len(df_hourly_from_db)} rekordów godzinowych.")

        df_hourly_from_db['snow'] = 0.0
        _snow_depth_mm = 0.0
        df_hourly_from_db.sort_index(inplace=True)
        _calculated_snow_values = []
        for index, row in df_hourly_from_db.iterrows():
            _hourly_precip_water_eq = row['prcp']; _avg_hourly_temp = row['temp']
            _fresh_snow_mm = 0.0
            if _hourly_precip_water_eq > 0 and _avg_hourly_temp < TEMP_THRESHOLD_SNOW:
                _fresh_snow_mm = _hourly_precip_water_eq * WATER_TO_SNOW_RATIO
            _snow_depth_mm += _fresh_snow_mm
            _melt_mm_water_eq = 0.0
            if _snow_depth_mm > 0 and _avg_hourly_temp > 0:
                _melt_mm_water_eq = HOURLY_MELT_RATE_PER_DEG_ABOVE_FREEZING * _avg_hourly_temp
                _melt_snow_depth_mm = _melt_mm_water_eq * WATER_TO_SNOW_RATIO
                _snow_depth_mm -= _melt_snow_depth_mm
            _snow_depth_mm = max(0, _snow_depth_mm)
            _snow_depth_mm = min(_snow_depth_mm, MAX_ACCUMULATED_SNOW_DEPTH_MM)
            _calculated_snow_values.append(round(_snow_depth_mm, 2))
        df_hourly_from_db['snow'] = _calculated_snow_values
        print("    Zakończono symulację pokrywy śnieżnej dla danych z bazy.")

        df_hourly_from_db['coco'] = 1 # Placeholder
        df_hourly_from_db['weather_category'] = df_hourly_from_db['coco'].apply(aggregate_coco_FINAL_user_v2)
        df_hourly_from_db['year'] = df_hourly_from_db.index.year

        df_for_feature_engineering = df_hourly_from_db.copy()
        print(f"    Przygotowano {len(df_for_feature_engineering)} rekordów z bazy do dalszego przetwarzania (inżynieria cech).")

    except Exception as e:
        print(f"KRYTYCZNY BŁĄD podczas przetwarzania danych z bazy: {e}"); import traceback; traceback.print_exc(); exit()

else: # --- Oryginalna logika dla Meteostat (trening/test) ---
    print("  Pobieranie i wstępne przetwarzanie danych z Meteostat...")
    all_station_data_list = []
    _df_meteostat = None
    for station_id in target_station_ids: # Pętla wykona się raz dla Wiednia
        print(f"    Pobieranie danych dla stacji: {station_id} ({station_names.get(station_id, '')})...", end="")
        start_fetch_time = time.time()
        station_hourly_data = Hourly(station_id, data_fetch_start_date_meteostat, data_fetch_end_date_meteostat)
        station_data = station_hourly_data.fetch()
        fetch_duration = time.time() - start_fetch_time
        if station_data.empty: print(f" BRAK DANYCH Meteostat. Przerywanie."); exit()
        print(f" Pobrano {len(station_data)} rek. w {fetch_duration:.1f}s.")
        
        required_cols = ['temp', 'rhum', 'coco', 'pres', 'wspd', 'prcp']
        optional_cols = ['tsun', 'wpgt', 'snow']
        missing_req = [col for col in required_cols if col not in station_data.columns]
        if missing_req: raise ValueError(f"Brak wymaganych kolumn z Meteostat: {', '.join(missing_req)}")
        
        for col in optional_cols:
            if col not in station_data.columns: station_data[col] = 0.0
            else: station_data[col] = station_data[col].fillna(0)
        station_data['prcp'] = station_data['prcp'].fillna(0)
        station_data.dropna(subset=['temp', 'rhum', 'coco', 'pres', 'wspd'], inplace=True)
        if station_data.empty: raise ValueError(f"Brak danych Meteostat po usunięciu NaN dla stacji {station_id}.")
        station_data = station_data[station_data['coco'] != 0]
        station_data['coco'] = station_data['coco'].astype(int)
        all_station_data_list.append(station_data)
        print(f"      Przetworzono dane Meteostat ze stacji {station_id}.")
    
    _df_meteostat = pd.concat(all_station_data_list)
    print(f"    DataFrame Meteostat zawiera {len(_df_meteostat)} rekordów.")
    print("    Sortowanie danych Meteostat wg czasu..."); _df_meteostat.sort_index(inplace=True)
    
    print("    Agregowanie kategorii Meteostat (user_v2)...")
    _df_meteostat['weather_category'] = _df_meteostat['coco'].apply(aggregate_coco_FINAL_user_v2)
    _df_meteostat = _df_meteostat[_df_meteostat['weather_category'] != 'Unknown']
    
    print("    Filtrowanie prcp=0 dla opadów (Meteostat)...")
    initial_rows_before_prcp_filter = len(_df_meteostat)
    condition_to_remove = (_df_meteostat['prcp'] == 0) & (_df_meteostat['weather_category'].isin(precip_categories_user))
    rows_to_remove_count = condition_to_remove.sum()
    if rows_to_remove_count > 0: _df_meteostat = _df_meteostat[~condition_to_remove].copy()
    print(f"      Usunięto {rows_to_remove_count} wierszy.")
    if _df_meteostat['weather_category'].nunique() < 2: raise ValueError("Mniej niż 2 kategorie po filtrowaniu danych Meteostat.")
    
    _df_meteostat['year'] = _df_meteostat.index.year # Dodanie kolumny 'year'
    df_for_feature_engineering = _df_meteostat.copy()
    print(f"  Przygotowano {len(df_for_feature_engineering)} rekordów z Meteostat do dalszego przetwarzania.")


if df_for_feature_engineering is None or df_for_feature_engineering.empty:
    print("KRYTYCZNY BŁĄD: DataFrame do inżynierii cech jest pusty. Przerywanie."); exit()


# --- Etap 2: Rozszerzona Inżynieria Cech v2 ---
# Ten blok operuje na `df_for_feature_engineering` i zapisuje wynik do `df_processed_final`
print("\n--- Etap 2: Rozszerzona Inżynieria Cech v2 ---")
feature_engineering_start_time_actual = time.time() # Zmieniona nazwa zmiennej, żeby nie kolidowała
df = df_for_feature_engineering.copy() # Używamy nazwy 'df' tak jak w oryginalnym bloku FE
epsilon = 1e-6

# 1. Cechy Podstawowe i Czasowe
print("  Tworzenie cech podstawowych i czasowych...")
df['dew_point'] = df.apply(lambda row: calculateDewPoint(row['temp'], row['rhum']), axis=1)
df['spread'] = df['temp'] - df['dew_point']
df['hour'] = df.index.hour
df['day_of_year'] = df.index.dayofyear
df['month'] = df.index.month
# 'year' już powinno być w df
df['day_of_week'] = df.index.dayofweek
df['week_of_year'] = df.index.isocalendar().week.astype(int)
df['quarter'] = df.index.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/366.0)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/366.0)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
if 'tsun' in df.columns and pd.api.types.is_numeric_dtype(df['tsun']): # Dodano sprawdzenie typu
    df['is_daytime_approx'] = (df['tsun'] > 0).astype(int)
else:
    df['is_daytime_approx'] = 0 # Jeśli tsun nie istnieje lub nie jest numeryczne

# 2. Różnice Czasowe
print("  Tworzenie cech różnic czasowych...")
periods_diff = [1, 2, 3, 6, 12, 24]
cols_to_diff = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'prcp', 'tsun', 'wpgt', 'snow']
diff_feature_names = []
for period in periods_diff:
    for col in cols_to_diff:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): # Sprawdzenie czy kolumna jest numeryczna
            diff_col_name = f'{col}_diff_{period}h'
            abs_diff_col_name = f'abs_{col}_diff_{period}h'
            df[diff_col_name] = df[col].diff(periods=period)
            df[abs_diff_col_name] = df[diff_col_name].abs()
            diff_feature_names.extend([diff_col_name, abs_diff_col_name])
        elif col in df.columns:
            print(f"    Ostrzeżenie: Kolumna '{col}' do różnicowania nie jest numeryczna i zostanie pominięta.")
        # else: # Kolumna nie istnieje, pomijamy po cichu
        #    pass


# 3. Wartości Opóźnione
print("  Tworzenie cech opóźnionych...")
periods_lag = [1, 2, 3, 6, 12, 24]
# Dodajemy nowo utworzone diff_1h do listy cech do opóźniania
# Upewnijmy się, że bierzemy tylko te diff_1h, które faktycznie zostały utworzone i są numeryczne
valid_diff_1h_features = [f_name for f_name in diff_feature_names if '_diff_1h' in f_name and f_name in df.columns and pd.api.types.is_numeric_dtype(df[f_name])]
cols_to_lag = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'prcp', 'tsun', 'wpgt', 'snow'] + valid_diff_1h_features
lagged_feature_names = []
for period in periods_lag:
    for col in cols_to_lag:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): # Sprawdzenie czy kolumna jest numeryczna
            lag_col_name = f'{col}_lag_{period}h'
            df[lag_col_name] = df[col].shift(periods=period)
            lagged_feature_names.append(lag_col_name)
        elif col in df.columns:
             print(f"    Ostrzeżenie: Kolumna '{col}' do opóźniania nie jest numeryczna i zostanie pominięta.")
        # else: # Kolumna nie istnieje (np. diff_1h nie powstał), pomijamy
        #    pass


# 4. Opóźnione Flagi Kategorii
print("  Tworzenie opóźnionych flag kategorii...")
lagged_cat_feature_names = []

# Jeśli 'weather_category' nie istnieje (np. błąd wcześniej), stwórz placeholder
if 'weather_category' not in df.columns:
    print("    OSTRZEŻENIE: Brak kolumny 'weather_category' do tworzenia flag opóźnionych. Używam placeholdera 'Unknown'.")
    df['weather_category'] = "Unknown"

# ZMIENIONE: Użyj wszystkich możliwych kategorii zdefiniowanych globalnie,
# a nie tylko tych, które aktualnie występują w df['weather_category']
# (bo w trybie predykcji z placeholderem coco, df['weather_category'] może być stałe)
all_possible_user_categories = ['Clear/Fair', 'Cloudy/Overcast', 'Fog', 'Rain', 'Snow/Sleet/Freezing', 'Thunderstorm/Severe']
# lub jeśli masz `all_categories_user` zdefiniowane globalnie:
# all_possible_user_categories = all_categories_user # Upewnij się, że ta lista zawiera wszystkie 6 kategorii

print(f"    Tworzenie flag opóźnionych dla potencjalnych kategorii: {all_possible_user_categories}")
for lag in [1, 2, 3, 6]:
    shifted_cat = df['weather_category'].shift(lag) # To nadal bazuje na aktualnej (może być stałej) weather_category
    
    # Twórz flagi dla WSZYSTKICH zdefiniowanych kategorii użytkownika
    for cat_possible in all_possible_user_categories:
        safe_cat_name = cat_possible.replace('/', '_').replace(' ', '_').replace('-', '_').lower()
        flag_name = f'was_{safe_cat_name}_lag{lag}h'
        # Jeśli aktualna (przesunięta) kategoria to `cat_possible`, flaga = 1, inaczej 0.
        # Nawet jeśli `shifted_cat` jest zawsze 'Clear/Fair', to dla `cat_possible` = 'Rain',
        # warunek `(shifted_cat == cat_possible)` będzie False, więc flaga `was_rain_lagXh` będzie 0.
        # To jest poprawne zachowanie.
        df[flag_name] = (shifted_cat == cat_possible).astype(int)
        lagged_cat_feature_names.append(flag_name)
        
    # Flaga dla kategorii opadowych - pozostaje bez zmian, bazuje na shifted_cat
    precip_flag_name = f'was_precip_category_lag{lag}h'
    # precip_categories_user powinno być zdefiniowane globalnie
    df[precip_flag_name] = shifted_cat.isin(precip_categories_user).astype(int)
    lagged_cat_feature_names.append(precip_flag_name)

# 5. Statystyki Kroczące
print("  Tworzenie statystyk kroczących...")
window_sizes = [3, 6, 12, 24]
cols_for_rolling = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'prcp', 'tsun', 'wpgt', 'snow'] # Dodano snow
rolling_feature_names = []
for window in window_sizes:
    for col in cols_for_rolling:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            rolling_window = df[col].rolling(window=window, closed='right', min_periods=max(1, window // 2))
            ops = {'mean': rolling_window.mean, 'std': rolling_window.std,
                   'median': rolling_window.median, 'min': rolling_window.min, 'max': rolling_window.max}
            if col in ['prcp', 'tsun']:
                ops['sum'] = rolling_window.sum
            for op_name, op_func in ops.items():
                feat_name = f'{col}_roll{window}h_{op_name}'
                df[feat_name] = op_func()
                rolling_feature_names.append(feat_name)
        elif col in df.columns:
            print(f"    Ostrzeżenie: Kolumna '{col}' dla statystyk kroczących nie jest numeryczna i zostanie pominięta.")


# 6. Interakcje i Cechy Pochodne
print("  Tworzenie interakcji i cech pochodnych...")
derived_feature_names = []
base_cols = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'tsun', 'wpgt'] # snow można dodać
for i in range(len(base_cols)):
    for j in range(i, len(base_cols)):
        col1, col2 = base_cols[i], base_cols[j]
        # Sprawdzenie czy obie kolumny istnieją i są numeryczne
        if col1 in df.columns and pd.api.types.is_numeric_dtype(df[col1]) and \
           col2 in df.columns and pd.api.types.is_numeric_dtype(df[col2]):
            if f'{col1}_x_{col2}' not in df.columns: df[f'{col1}_x_{col2}'] = df[col1] * df[col2]; derived_feature_names.append(f'{col1}_x_{col2}')
            if f'{col1}_div_{col2}_safe' not in df.columns: df[f'{col1}_div_{col2}_safe'] = df[col1] / (df[col2] + epsilon); derived_feature_names.append(f'{col1}_div_{col2}_safe')
            if i != j and f'{col2}_div_{col1}_safe' not in df.columns: df[f'{col2}_div_{col1}_safe'] = df[col2] / (df[col1] + epsilon); derived_feature_names.append(f'{col2}_div_{col1}_safe')

cols_for_pow = base_cols + ['prcp', 'snow'] # Dodano snow
for col in cols_for_pow:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        if f'{col}_pow2' not in df.columns: df[f'{col}_pow2'] = df[col].pow(2); derived_feature_names.append(f'{col}_pow2')
        if col in ['wspd', 'spread', 'prcp', 'snow'] and f'{col}_pow3' not in df.columns: df[f'{col}_pow3'] = df[col].pow(3); derived_feature_names.append(f'{col}_pow3')

# Sprawdzenia dla konkretnych interakcji
def check_and_create_interaction(df_ref, derived_list, new_feat_name, col_list, operation_str):
    # Sprawdza czy wszystkie potrzebne kolumny istnieją i są numeryczne
    valid_cols = all(c in df_ref.columns and pd.api.types.is_numeric_dtype(df_ref[c]) for c in col_list)
    if valid_cols:
        try:
            # Używamy eval do wykonania operacji, wymaga ostrożności
            # Alternatywnie, można użyć if/else dla każdej operacji
            df_ref[new_feat_name] = eval(operation_str, {'df': df_ref, 'epsilon': epsilon, 'abs': abs, 'np': np})
            derived_list.append(new_feat_name)
        except Exception as e:
            print(f"    Błąd przy tworzeniu cechy interakcji '{new_feat_name}': {e}")
    # else:
    #    print(f"    Pominięto tworzenie '{new_feat_name}' z powodu braku/niepoprawnego typu kolumn: {col_list}")


check_and_create_interaction(df, derived_feature_names, 'abs_temp_diff_div_wspd_safe', ['temp_diff_1h', 'wspd'], "abs(df['temp_diff_1h']) / (df['wspd'] + epsilon)")
check_and_create_interaction(df, derived_feature_names, 'abs_pres_diff_div_wspd_safe', ['pres_diff_1h', 'wspd'], "abs(df['pres_diff_1h']) / (df['wspd'] + epsilon)")
check_and_create_interaction(df, derived_feature_names, 'temp_x_abs_pres_diff_1h', ['temp', 'pres_diff_1h'], "df['temp'] * abs(df['pres_diff_1h'])")
check_and_create_interaction(df, derived_feature_names, 'rhum_x_abs_spread_diff_1h', ['rhum', 'spread_diff_1h'], "df['rhum'] * abs(df['spread_diff_1h'])")
check_and_create_interaction(df, derived_feature_names, 'prcp_lag_1h_x_temp', ['prcp_lag_1h', 'temp'], "df['prcp_lag_1h'] * df['temp']")

for window in window_sizes:
    for col in ['temp', 'rhum', 'spread', 'pres', 'wspd']:
        mean_col_name = f'{col}_roll{window}h_mean'
        rel_col_name = f'{col}_rel_to_roll{window}h_mean'
        check_and_create_interaction(df, derived_feature_names, rel_col_name, [col, mean_col_name], f"df['{col}'] - df['{mean_col_name}']")


# 7. Dodatkowe Cechy Matematyczne
print("  Tworzenie dodatkowych cech matematycznych...")
additional_math_features = []
cols_for_adv_math = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd'] # snow można dodać
for window in window_sizes:
    for col in cols_for_adv_math:
        min_col=f'{col}_roll{window}h_min'; max_col=f'{col}_roll{window}h_max'; range_col=f'{col}_roll{window}h_range'
        std_col=f'{col}_roll{window}h_std'; ratio_std_col=f'{col}_div_roll{window}h_std_safe'
        diff_1h_col=f'{col}_diff_1h'; volatility_col=f'{diff_1h_col}_roll{window}h_std'
        
        check_and_create_interaction(df, additional_math_features, range_col, [min_col, max_col], f"df['{max_col}'] - df['{min_col}']")
        check_and_create_interaction(df, additional_math_features, ratio_std_col, [col, std_col], f"df['{col}'] / (df['{std_col}'] + epsilon)")
        
        if diff_1h_col in df.columns and pd.api.types.is_numeric_dtype(df[diff_1h_col]):
            df[volatility_col] = df[diff_1h_col].rolling(window=window, min_periods=max(1, window//2)).std()
            additional_math_features.append(volatility_col)

for col in cols_for_adv_math:
    diff_1h_col = f'{col}_diff_1h'; accel_col = f'{diff_1h_col}_diff_1h'
    if diff_1h_col in df.columns and pd.api.types.is_numeric_dtype(df[diff_1h_col]):
        df[accel_col] = df[diff_1h_col].diff(1)
        additional_math_features.append(accel_col)

check_and_create_interaction(df, additional_math_features, 'dp_div_temp_safe', ['dew_point', 'temp'], "df['dew_point'] / (df['temp'] + np.sign(df['temp'])*epsilon + epsilon)")
check_and_create_interaction(df, additional_math_features, 'spread_div_temp_safe', ['spread', 'temp'], "df['spread'] / (df['temp'] + np.sign(df['temp'])*epsilon + epsilon)")
check_and_create_interaction(df, additional_math_features, 'temp_x_hour_sin', ['temp', 'hour_sin'], "df['temp'] * df['hour_sin']")
check_and_create_interaction(df, additional_math_features, 'wspd_x_day_of_year_cos', ['wspd', 'day_of_year_cos'], "df['wspd'] * df['day_of_year_cos']")
check_and_create_interaction(df, additional_math_features, 'rhum_x_pres_diff_1h', ['rhum', 'pres_diff_1h'], "df['rhum'] * df['pres_diff_1h']")


# 8. Rozszerzone Flagi Binarne
print("  Obliczanie flag binarnych...")
thresholds = {
    'flag_sunny': ('tsun', '>', 45), 'flag_partly_sunny': ('tsun', '>', 15), 'flag_mostly_dark': ('tsun', '<', 10), 'flag_dark': ('tsun', '<', 1),
    'flag_very_humid': ('rhum', '>', 96), 'flag_humid': ('rhum', '>', 85), 'flag_moderate_humid': ('rhum', '<=', 85), 'flag_dry': ('rhum', '<', 65), 'flag_very_dry': ('rhum', '<', 45),
    'flag_near_saturation': ('spread', '<', 0.7), 'flag_close_saturation': ('spread', '<', 1.5), 'flag_moderate_spread': ('spread', '>', 3.0), 'flag_far_from_saturation': ('spread', '>', 6.0),
    'flag_calm': ('wspd', '<', 2.5), 'flag_light_breeze': ('wspd', '>', 5.0), 'flag_moderate_wind': ('wspd', '>', 8.0), 'flag_windy': ('wspd', '>', 12.0), 'flag_very_windy': ('wspd', '>', 18.0),
    'flag_has_precip': ('prcp', '>', 0), 'flag_light_precip': ('prcp', '>', 0.1), 'flag_moderate_precip': ('prcp', '>', 1.0), 'flag_heavy_precip': ('prcp', '>', 3.0),
    'flag_very_high_pres': ('pres', '>', 1030), 'flag_high_pres': ('pres', '>', 1020), 'flag_moderate_pres': ('pres', '<=', 1020), 'flag_low_pres': ('pres', '<', 1005), 'flag_very_low_pres': ('pres', '<', 990),
    'flag_below_freezing': ('temp', '<=', 0), 'flag_near_freezing_low': ('temp', '<=', 2), 'flag_near_freezing_high': ('temp', '<=', 5), 'flag_cool': ('temp', '<', 10), 'flag_mild': ('temp', '>=', 10), 'flag_warm': ('temp', '>', 18), 'flag_hot': ('temp', '>', 25),
    'flag_dp_below_freezing': ('dew_point', '<=', 0), 'flag_dp_low': ('dew_point', '<', 5), 'flag_dp_high': ('dew_point', '>', 15), 'flag_dp_very_high': ('dew_point', '>', 20),
    'flag_has_gusts': ('wpgt', '>', 0), 'flag_strong_gusts': ('wpgt', '>', 15), 'flag_severe_gusts': ('wpgt', '>', 25),
    'flag_has_snow_cover': ('snow', '>', 0), 'flag_sig_snow_cover': ('snow', '>', 50), # Używa oszacowanego 'snow'
    'flag_temp_rising_fast': ('temp_diff_1h', '>', 1.5), 'flag_temp_falling_fast': ('temp_diff_1h', '<', -1.5),
    'flag_pres_rising_fast': ('pres_diff_1h', '>', 1.0), 'flag_pres_falling_fast': ('pres_diff_1h', '<', -1.0), 'flag_pres_falling_sig': ('pres_diff_1h', '<', -0.5),
    'flag_rhum_rising_fast': ('rhum_diff_1h', '>', 10), 'flag_rhum_falling_fast': ('rhum_diff_1h', '<', -10),
    'flag_spread_closing_fast': ('spread_diff_1h', '<', -0.8), 'flag_spread_opening_fast': ('spread_diff_1h', '>', 1.0),
    'flag_wspd_increasing': ('wspd_diff_1h', '>', 2.0), 'flag_wspd_decreasing': ('wspd_diff_1h', '<', -2.0),
    'flag_fog_ratio': ('rhum_div_spread_safe', '>', 120), 'flag_clear_ratio_rhum_spread': ('rhum_div_spread_safe', '<', 20), 'flag_clear_ratio_tsun_rhum': ('tsun_div_rhum_safe', '>', 0.6),
    'flag_high_temp_range_6h': ('temp_roll6h_range', '>', 5), 'flag_low_pres_range_12h': ('pres_roll12h_range', '<', 5),
    'flag_temp_accel_positive': ('temp_diff_1h_diff_1h', '>', 0.5), 'flag_pres_accel_negative': ('pres_diff_1h_diff_1h', '<', -0.3),
    'flag_high_rhum_volatility': ('rhum_diff_1h_roll6h_std', '>', 8),
    'flag_cold_and_humid': (('temp', '<', 5), ('rhum', '>', 90)), 'flag_warm_and_dry': (('temp', '>', 20), ('rhum', '<', 50)),
    'flag_windy_and_precip': (('flag_windy', '==', 1), ('flag_has_precip', '==', 1)), 'flag_gusty_and_precip': (('flag_strong_gusts', '==', 1), ('flag_has_precip', '==', 1)),
    'flag_near_freezing_precip': (('flag_near_freezing_low', '==', 1), ('flag_has_precip', '==', 1)), 'flag_below_freezing_precip': (('flag_below_freezing', '==', 1), ('flag_has_precip', '==', 1)),
    'flag_fog_conditions_met': (('flag_near_saturation', '==', 1), ('flag_calm', '==', 1), ('flag_very_humid', '==', 1)),
    'flag_potential_thunder_convection': (('temp', '>', 15), ('spread', '<', 8), ('flag_pres_falling_sig', '==', 1)),
    'flag_potential_snow': (('flag_near_freezing_low', '==', 1), ('flag_dp_below_freezing', '==', 1), ('flag_has_precip', '==', 1)),
    'flag_saturating_trend': (('flag_rhum_rising_fast', '==', 1), ('flag_spread_closing_fast', '==', 1)),
    'flag_damp_conditions': (('rhum', '>', 92), ('spread', '<', 1.5)), 'flag_frontal_drizzle_potential': (('flag_pres_falling_sig', '==', 1), ('rhum', '>', 85), ('spread', '<', 3.0)),
}
calculated_flags = set()
flags_to_calculate = list(thresholds.keys())
max_iterations = 5
iteration = 0
threshold_flags_names = [] # Lista do przechowywania nazw utworzonych flag
thresholds_copy = thresholds.copy() # Kopia do bezpiecznego usuwania

while flags_to_calculate and iteration < max_iterations:
    newly_calculated_in_iter = [] # Flagi obliczone w tej iteracji
    remaining_flags_for_next_iter = []
    
    for flag_name in list(flags_to_calculate): # Iterujemy po kopii listy, aby móc ją modyfikować
        if flag_name not in thresholds_copy: # Jeśli flaga została usunięta z powodu błędu
            if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name) # Upewnij się, że jest usunięta
            continue

        conditions = thresholds_copy[flag_name]
        can_calculate_flag = True
        
        current_required_features_for_flag = []
        is_combined_flag = isinstance(conditions[0], tuple)
        if is_combined_flag:
            current_required_features_for_flag = [cond[0] for cond in conditions]
        else:
            current_required_features_for_flag = [conditions[0]]

        for feature_needed in current_required_features_for_flag:
            is_dependency_on_other_flag = feature_needed.startswith('flag_')
            if is_dependency_on_other_flag and feature_needed not in calculated_flags:
                can_calculate_flag = False; break
            elif not is_dependency_on_other_flag and (feature_needed not in df.columns or not pd.api.types.is_numeric_dtype(df[feature_needed])):
                print(f"    BŁĄD KRYTYCZNY: Bazowa cecha '{feature_needed}' (numeryczna) dla flagi '{flag_name}' nie istnieje lub nie jest numeryczna. Pomijam flagę '{flag_name}'.")
                if flag_name in thresholds_copy: del thresholds_copy[flag_name]
                if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name)
                can_calculate_flag = False; break
        
        if not can_calculate_flag:
            if flag_name in thresholds_copy: # Jeśli błąd nie był krytyczny (tylko zależność)
                 remaining_flags_for_next_iter.append(flag_name)
            continue

        # Obliczanie flagi
        final_condition_series = pd.Series(True, index=df.index)
        process_condition_lambda = lambda feat, op, thr: {
            '<': df[feat] < thr, '>': df[feat] > thr,
            '<=': df[feat] <= thr, '>=': df[feat] >= thr,
            '==': df[feat] == thr, '!=': df[feat] != thr
        }.get(op, pd.Series(False, index=df.index)) # Fallback na False

        try:
            if is_combined_flag:
                for feature, operator, threshold_val in conditions:
                    if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]): # Podwójne sprawdzenie
                        raise KeyError(f"Cecha '{feature}' dla flagi '{flag_name}' nie istnieje lub nie jest numeryczna w df podczas budowania warunku.")
                    final_condition_series &= process_condition_lambda(feature, operator, threshold_val)
            else:
                feature, operator, threshold_val = conditions
                if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]): # Podwójne sprawdzenie
                    raise KeyError(f"Cecha '{feature}' dla flagi '{flag_name}' nie istnieje lub nie jest numeryczna w df podczas budowania warunku.")
                final_condition_series = process_condition_lambda(feature, operator, threshold_val)
            
            df[flag_name] = final_condition_series.astype(int)
            threshold_flags_names.append(flag_name)
            calculated_flags.add(flag_name)
            newly_calculated_in_iter.append(flag_name)
            if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name)
            if flag_name in thresholds_copy: del thresholds_copy[flag_name]

        except KeyError as ke_flag:
            print(f"    BŁĄD KRYTYCZNY (KeyError) przy obliczaniu flagi '{flag_name}': {ke_flag}. Pomijam flagę.")
            if flag_name in thresholds_copy: del thresholds_copy[flag_name]
            if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name)
        except Exception as e_flag:
            print(f"    BŁĄD (Inny) przy obliczaniu flagi '{flag_name}': {e_flag}. Spróbuję później.")
            if flag_name not in remaining_flags_for_next_iter and flag_name in thresholds_copy:
                 remaining_flags_for_next_iter.append(flag_name)

    flags_to_calculate = remaining_flags_for_next_iter
    iteration += 1
    if not newly_calculated_in_iter and flags_to_calculate:
        print(f"    OSTRZEŻENIE: W iteracji {iteration} nie udało się obliczyć żadnych nowych flag. Pozostałe flagi do obliczenia: {flags_to_calculate}")
        unresolved_dependencies_report = {}
        for fname_report in flags_to_calculate:
            if fname_report in thresholds_copy:
                conds_report = thresholds_copy[fname_report]
                is_comb_report = isinstance(conds_report[0], tuple)
                req_feats_report = [c[0] for c in conds_report] if is_comb_report else [conds_report[0]]
                missing_deps_report = [rf for rf in req_feats_report if rf.startswith('flag_') and rf not in calculated_flags]
                if missing_deps_report: unresolved_dependencies_report[fname_report] = missing_deps_report
        if unresolved_dependencies_report: print(f"      Nierozwiązane zależności flag: {unresolved_dependencies_report}")
        break 
    
if flags_to_calculate:
    print(f"    OSTRZEŻENIE KOŃCOWE: Nie udało się obliczyć wszystkich flag po {max_iterations} iteracjach. Nieuobliczone: {flags_to_calculate}")
print(f"  Utworzono {len(threshold_flags_names)} flag binarnych.")
# Koniec bloku flag binarnych

feature_engineering_duration_actual = time.time() - feature_engineering_start_time_actual
print(f"--- Zakończono Rozszerzoną Inżynierię Cech v2 ({feature_engineering_duration_actual:.1f} sek) ---")

# --- FINALNE CZYSZCZENIE NaN ---
print("\n  Usuwanie NaN po pełnej inżynierii cech (finalne)...")
rows_before_final_dropna = len(df)
# Upewnij się, że 'year' i 'coco' są w df, jeśli używasz ich w cols_to_exclude_from_dropna
if 'year' not in df.columns: df['year'] = df.index.year
if 'coco' not in df.columns: df['coco'] = 1 # Placeholder jeśli brak

potential_feature_cols = list(df.select_dtypes(include=np.number).columns)
cols_to_exclude_from_dropna = ['coco', 'year'] # Te kolumny nie są cechami numerycznymi do modelu
# Tworzymy listę cech, na podstawie których będziemy usuwać NaN
# Powinny to być wszystkie numeryczne cechy, które mogą być użyte w modelach
features_for_dropna_final = [f for f in potential_feature_cols if f not in cols_to_exclude_from_dropna and f in df.columns]

if not features_for_dropna_final:
    print("  OSTRZEŻENIE: Brak numerycznych cech (poza 'coco', 'year') do sprawdzenia NaN. Nie wykonano dropna.")
else:
    # Usuwamy wiersze, które mają NaN w którejkolwiek z wybranych cech numerycznych
    # To jest ważne, bo XGBoost nie lubi NaN.
    df.dropna(subset=features_for_dropna_final, inplace=True)

rows_after_processing_final = len(df)
print(f"  Usunięto {rows_before_final_dropna - rows_after_processing_final} wierszy z NaN.")
print(f"  Ostateczna liczba rekordów po inżynierii cech: {rows_after_processing_final}.")

if rows_after_processing_final == 0:
    print("KRYTYCZNY BŁĄD: Brak danych po pełnym przetworzeniu i usunięciu NaN. Sprawdź logikę i dane wejściowe."); exit()

df_processed_final = df.copy() # Wynik inżynierii cech
# --- KONIEC ETAPU 2 ---

processing_and_fe_duration = time.time() - full_processing_start_time
print(f"--- Całkowity czas przygotowania danych i FE: {processing_and_fe_duration:.1f} sek ---")


# --- Definicje List Cech z ANOVA v8 (Wiedeń) ---
# UPEWNIJ SIĘ, ŻE TE LISTY SĄ IDENTYCZNE JAK W SKRYPCIE TRENINGOWYM
print("\n--- Etap 3: Definiowanie list cech ---")
FEATURES_M1 = [
    'was_precip_category_lag1h', 'flag_light_precip', 'flag_has_precip', 'flag_gusty_and_precip',
    'was_rain_lag1h', 'flag_windy_and_precip', 'was_precip_category_lag2h', 'was_rain_lag2h',
    'was_precip_category_lag3h', 'was_rain_lag3h', 'prcp_roll6h_median', 'prcp_roll3h_sum',
    'prcp_roll3h_mean', 'prcp_roll6h_mean', 'prcp_roll6h_sum', 'prcp_roll3h_median',
    'prcp_roll3h_min', 'prcp_roll12h_mean', 'prcp_roll12h_sum', 'prcp_roll12h_median',
    'prcp_roll3h_max', 'prcp', 'prcp_roll6h_min', 'flag_moderate_precip',
    'prcp_roll6h_max', 'prcp_lag_1h', 'was_precip_category_lag6h', 'prcp_roll6h_std',
    'prcp_roll3h_std', 'flag_near_freezing_precip'
]
FEATURES_M2 = [
    'was_fog_lag1h', 'flag_damp_conditions', 'was_fog_lag2h', 'flag_fog_ratio',
    'flag_near_saturation', 'flag_close_saturation', 'was_fog_lag3h', 'flag_very_humid',
    'flag_cold_and_humid', 'flag_moderate_humid', 'flag_humid', 'rhum_x_spread',
    'was_fog_lag6h', 'rhum_pow2', 'rhum_x_rhum', 'flag_moderate_spread',
    'rhum_roll6h_min', 'rhum_div_roll24h_std_safe', 'rhum_roll3h_min', 'rhum_roll12h_min',
    'rhum_x_pres', 'rhum', 'rhum_roll3h_mean', 'rhum_roll24h_min',
    'rhum_div_pres_safe', 'rhum_roll24h_mean', 'rhum_roll6h_mean', 'rhum_roll3h_median',
    'rhum_roll12h_mean', 'rhum_lag_1h'
]
FEATURES_M3 = [
    'flag_potential_snow', 'flag_dp_below_freezing', 'flag_near_freezing_precip', 'flag_near_freezing_low',
    'was_snow_sleet_freezing_lag1h', 'flag_below_freezing', 'flag_below_freezing_precip', 'was_snow_sleet_freezing_lag2h',
    'was_snow_sleet_freezing_lag3h', 'temp_x_rhum', 'temp_div_pres_safe', 'temp',
    'temp_x_pres', 'temp_roll3h_min', 'temp_roll3h_mean', 'dew_point_roll3h_max',
    'dew_point_roll6h_max', 'temp_roll3h_median', 'dew_point_roll3h_mean', 'dew_point_roll12h_max',
    'temp_lag_1h', 'dew_point',
    'dew_point_div_pres_safe', 'dew_point_x_pres',
    'dew_point_roll3h_median', 'dew_point_lag_1h', 'dew_point_roll24h_max', 'temp_roll3h_max',
    'temp_roll6h_min', 'rhum_x_dew_point'
]
FEATURES_M4 = [
    'was_clear_fair_lag1h', 'was_cloudy_overcast_lag1h', 'was_clear_fair_lag2h', 'was_cloudy_overcast_lag2h',
    'was_clear_fair_lag3h', 'was_cloudy_overcast_lag3h', 'was_clear_fair_lag6h', 'was_cloudy_overcast_lag6h',
    'temp_roll12h_range', 'tsun_roll24h_sum', 'tsun_roll24h_mean', 'temp_roll12h_std',
    'temp_roll24h_std', 'temp_roll24h_range', 'tsun_roll12h_sum', 'tsun_roll12h_mean',
    'spread_roll24h_std', 'spread_roll12h_max', 'rhum_roll12h_min', 'rhum_roll24h_min',
    'spread_roll12h_range', 'spread_roll24h_range', 'spread_roll24h_max', 'spread_roll12h_std',
    'tsun_roll12h_max', 'tsun_roll12h_std', 'abs_tsun_diff_12h', 'rhum_roll24h_std',
    'tsun_roll24h_std', 'tsun_roll6h_mean'
]

print("  Weryfikacja dostępności cech...")
feature_lists_final = {}
all_features_unpacked_for_models = [] # Zmieniona nazwa, aby uniknąć konfliktu
for name, features_list_for_model in [("M1", FEATURES_M1), ("M2", FEATURES_M2), ("M3", FEATURES_M3), ("M4", FEATURES_M4)]:
    available_model_features = [f for f in features_list_for_model if f in df_processed_final.columns]
    missing_model_features = sorted(list(set(features_list_for_model) - set(available_model_features)))
    
    non_flag_missing_model = [m for m in missing_model_features if not m.startswith('flag_')]
    # threshold_flags_names jest zdefiniowane w bloku FE
    flag_missing_not_created_model = [m for m in missing_model_features if m.startswith('flag_') and m not in threshold_flags_names]

    if non_flag_missing_model:
         print(f"    KRYTYCZNE OSTRZEŻENIE ({name}): Nie znaleziono NIE-FLAGOWYCH cech: {', '.join(non_flag_missing_model)}.")
    if flag_missing_not_created_model:
         print(f"    OSTRZEŻENIE ({name}): Flagi nie znalezione LUB nie utworzone: {', '.join(flag_missing_not_created_model)}.")
    
    if missing_model_features:
        print(f"    ({name}): Usunięto {len(missing_model_features)} brakujących/nieutworzonych cech z listy dla tego modelu.")
        current_model_features_final = available_model_features
    else:
        # print(f"    ({name}): Wszystkie {len(features_list_for_model)} cechy są dostępne.") # Mniej gadatliwe
        current_model_features_final = features_list_for_model

    if not current_model_features_final:
        print(f"    KRYTYCZNY BŁĄD ({name}): Brak dostępnych cech po weryfikacji! Model nie może być użyty.")
        feature_lists_final[name] = [] # Pusta lista spowoduje pominięcie modelu
    else:
        feature_lists_final[name] = current_model_features_final
        all_features_unpacked_for_models.extend(current_model_features_final)

all_unique_features_needed_by_models = sorted(list(set(all_features_unpacked_for_models)))
print(f"  Łącznie unikalnych cech potrzebnych przez wszystkie modele (po weryfikacji): {len(all_unique_features_needed_by_models)}")


# --- Etap 4: Przygotowanie Danych do Predykcji / Treningu i Testu ---
print("\n--- Etap 4: Przygotowanie Danych do Predykcji / Treningu i Testu ---")

if USE_DATABASE_INPUT:
    # Filtrujemy dane po inżynierii cech do zadanego okresu predykcji
    # Upewnij się, że indeks jest typu datetime
    if not isinstance(df_processed_final.index, pd.DatetimeIndex):
        df_processed_final.index = pd.to_datetime(df_processed_final.index)

    prediction_data_df = df_processed_final[
        (df_processed_final.index >= PREDICTION_START_DATE) &
        (df_processed_final.index <= PREDICTION_END_DATE)
    ].copy()
    print(f"  Przygotowano {len(prediction_data_df)} próbek do predykcji (z okresu {PREDICTION_START_DATE} - {PREDICTION_END_DATE}).")
    if prediction_data_df.empty:
        print("  BŁĄD: Brak danych w zadanym okresie predykcji po pełnym przetworzeniu. Sprawdź zakresy dat i logikę filtrowania.");
        exit()
    # W trybie predykcji nie mamy `y_test_actual_str` z góry, chyba że to re-predykcja dla ewaluacji
    # Dla czystej predykcji, ewaluacja końcowa nie będzie możliwa bez rzeczywistych etykiet.
else: # Tryb Meteostat (oryginalny podział na train/test)
    train_df = df_processed_final[(df_processed_final['year'] >= train_start_year) & (df_processed_final['year'] <= train_end_year)].copy()
    test_df = df_processed_final[df_processed_final['year'] == test_year].copy()
    print(f"  Liczba próbek treningowych (Meteostat, przed SMOTE): {len(train_df)}")
    print(f"  Liczba próbek testowych (Meteostat): {len(test_df)}")
    if train_df.empty or test_df.empty: print("  BŁĄD: Zbiór treningowy lub testowy Meteostat jest pusty."); exit()
    y_test_actual_str_meteostat = test_df['weather_category'] # Rzeczywiste etykiety dla danych testowych Meteostat


# --- Etap 5: Trening (jeśli FORCE_RETRAIN) lub Wczytywanie Modeli ---
# Funkcja pomocnicza do treningu (jeśli FORCE_RETRAIN) - skopiowana z Twojego skryptu
def train_xgboost_model(X_train, y_train, features, model_name_func, objective_func, num_class_func=None, use_smote_func=False, label_encoder_func=None):
    # ... (pełna definicja funkcji train_xgboost_model z Twojego skryptu)
    print(f"\n--- Trenowanie {model_name_func} ---")
    if not features: print(f"   BŁĄD: Brak cech dla {model_name_func}."); return None, None
    X_train_model = X_train[features].copy() # Użyj .copy()
    y_train_model = y_train.copy()
    if X_train_model.empty or len(y_train_model) == 0: print(f"   BŁĄD: Brak danych dla {model_name_func}."); return None, None
    for col in X_train_model.columns:
        if X_train_model[col].dtype == 'object':
            try: X_train_model[col] = pd.to_numeric(X_train_model[col])
            except ValueError: print(f"OSTRZEŻENIE: Nie udało się przekonwertować {col} na typ numeryczny.")
        if X_train_model[col].dtype == 'bool': X_train_model[col] = X_train_model[col].astype(int)
    print(f"   Rozkład klas przed SMOTE ({model_name_func}): {np.bincount(y_train_model) if len(np.unique(y_train_model)) > 0 else 'brak klas'}")
    X_train_resampled, y_train_resampled = X_train_model, y_train_model; scale_pos_weight_val = 1
    if use_smote_func and SMOTE_AVAILABLE:
        unique_classes, counts = np.unique(y_train_model, return_counts=True); min_class_count = counts.min() if len(counts)>0 else 0
        if len(unique_classes) > 1 and min_class_count >= 2: # Zmieniono na >=2 dla SMOTE
            k_neighbors_smote = min(5, min_class_count - 1) if min_class_count > 1 else 1
            if k_neighbors_smote < 1: k_neighbors_smote = 1
            print(f"   Stosowanie SMOTE (min klasa: {min_class_count}, k_neighbors={k_neighbors_smote})...")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            try: X_train_resampled, y_train_resampled = smote.fit_resample(X_train_model, y_train_model); print(f"   Rozkład klas PO SMOTE: {np.bincount(y_train_resampled)}")
            except ValueError as e: print(f"   OSTRZ.: Błąd SMOTE: {e}. Używam oryg. danych.")
        elif len(unique_classes) > 1:
            print(f"   OSTRZ.: Za mało próbek ({min_class_count}) dla SMOTE (wymagane min. 2).")
            if objective_func == 'binary:logistic' and len(counts) == 2 and counts[0] > 0 and counts[1] > 0 :
                 scale_pos_weight_val = counts[0] / counts[1]; print(f"   Używam scale_pos_weight = {scale_pos_weight_val:.2f}")
        elif len(unique_classes) <=1:
             print(f"   OSTRZ.: Tylko jedna klasa ({unique_classes}) w danych treningowych. SMOTE nie zostanie zastosowane.")
    elif not use_smote_func and objective_func == 'binary:logistic':
        counts = np.bincount(y_train_model)
        if len(counts) == 2 and counts[0] > 0 and counts[1] > 0: scale_pos_weight_val = counts[0] / counts[1]; print(f"   SMOTE wyłączone. Używam scale_pos_weight = {scale_pos_weight_val:.2f}")
    xgb_params = {'objective': objective_func, 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.7, 'colsample_bytree': 0.7, 'eval_metric': 'logloss' if 'binary' in objective_func else 'mlogloss', 'random_state': 42, 'n_jobs': -1}
    if objective_func == 'binary:logistic' and scale_pos_weight_val != 1: xgb_params['scale_pos_weight'] = scale_pos_weight_val
    if num_class_func: xgb_params['num_class'] = num_class_func
    model_xgb = xgb.XGBClassifier(**xgb_params); print(f"   Trenowanie XGBoost dla {model_name_func}..."); _start_time_train = time.time()
    try: model_xgb.fit(X_train_resampled, y_train_resampled); print(f"   Trening {model_name_func} zakończony w {time.time() - _start_time_train:.1f} sek."); return model_xgb, label_encoder_func
    except Exception as e_train: print(f"   BŁĄD treningu {model_name_func}: {e_train}"); return None, None


print("\n--- Etap 5: Trening lub Wczytywanie Modeli ---")
# Model 1
model_1_path = os.path.join(MODEL_SAVE_DIR, "model_M1.json"); model_1 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_1_path):
    print(f"  Wczytywanie Modelu 1 z: {model_1_path}")
    try: model_1 = xgb.XGBClassifier(); model_1.load_model(model_1_path); print("    Model 1 wczytany.")
    except Exception as e: print(f"    BŁĄD wczytywania M1: {e}."); model_1 = None
if (model_1 is None or FORCE_RETRAIN) and not USE_DATABASE_INPUT: # Trening tylko jeśli nie predykcja z bazy i trzeba
    print(f"  {'Wymuszono' if FORCE_RETRAIN else 'Rozpoczynam'} trening Modelu 1...")
    y_train_m1 = train_df['weather_category'].isin(precip_categories_user).astype(int)
    model_1, _ = train_xgboost_model(train_df, y_train_m1, feature_lists_final['M1'], "Model 1", 'binary:logistic', use_smote_func=USE_SMOTE)
    if model_1: 
        try: model_1.save_model(model_1_path); print(f"    Model 1 zapisany: {model_1_path}")
        except Exception as e: print(f"    BŁĄD zapisu M1: {e}")
elif model_1 is None and USE_DATABASE_INPUT: print("  KRYTYCZNY BŁĄD: Model 1 nie został wczytany, a jest potrzebny do predykcji."); exit()

# Model 2
model_2_path = os.path.join(MODEL_SAVE_DIR, "model_M2.json"); model_2 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_2_path):
    print(f"  Wczytywanie Modelu 2 z: {model_2_path}")
    try: model_2 = xgb.XGBClassifier(); model_2.load_model(model_2_path); print("    Model 2 wczytany.")
    except Exception as e: print(f"    BŁĄD wczytywania M2: {e}."); model_2 = None
if (model_2 is None or FORCE_RETRAIN) and not USE_DATABASE_INPUT:
    print(f"  {'Wymuszono' if FORCE_RETRAIN else 'Rozpoczynam'} trening Modelu 2...")
    train_df_m2_subset = train_df[train_df['weather_category'].isin(no_precip_categories_user)].copy()
    y_train_m2 = (train_df_m2_subset['weather_category'] == 'Fog').astype(int)
    model_2, _ = train_xgboost_model(train_df_m2_subset, y_train_m2, feature_lists_final['M2'], "Model 2", 'binary:logistic', use_smote_func=USE_SMOTE)
    if model_2: 
        try: model_2.save_model(model_2_path); print(f"    Model 2 zapisany: {model_2_path}")
        except Exception as e: print(f"    BŁĄD zapisu M2: {e}")
elif model_2 is None and USE_DATABASE_INPUT: print("  KRYTYCZNY BŁĄD: Model 2 nie został wczytany."); exit()

# Model 3 i LabelEncoder
model_3_path = os.path.join(MODEL_SAVE_DIR, "model_M3.json"); le_3_path = os.path.join(MODEL_SAVE_DIR, "le_M3.pkl")
model_3 = None; le_precip_trained_for_m3 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_3_path) and os.path.exists(le_3_path):
    print(f"  Wczytywanie Modelu 3 i LabelEncodera...")
    try:
        model_3 = xgb.XGBClassifier(); model_3.load_model(model_3_path)
        le_precip_trained_for_m3 = joblib.load(le_3_path)
        print(f"    Model 3 wczytany. LE dla M3 wczytany. Klasy: {list(le_precip_trained_for_m3.classes_)}")
    except Exception as e: print(f"    BŁĄD wczytywania M3/LE: {e}."); model_3 = None; le_precip_trained_for_m3 = None
if (model_3 is None or le_precip_trained_for_m3 is None or FORCE_RETRAIN) and not USE_DATABASE_INPUT:
    print(f"  {'Wymuszono' if FORCE_RETRAIN else 'Rozpoczynam'} trening Modelu 3...")
    train_df_m3_subset = train_df[train_df['weather_category'].isin(precip_categories_user)].copy()
    if not train_df_m3_subset.empty:
        current_le_m3 = LabelEncoder()
        y_train_m3 = current_le_m3.fit_transform(train_df_m3_subset['weather_category'])
        num_classes_m3 = len(current_le_m3.classes_)
        print(f"    Mapowanie klas M3 (trening): {dict(zip(current_le_m3.classes_, range(num_classes_m3)))}")
        model_3, le_precip_trained_for_m3 = train_xgboost_model(train_df_m3_subset, y_train_m3, feature_lists_final['M3'], "Model 3", 'multi:softmax', num_class_func=num_classes_m3, use_smote_func=USE_SMOTE, label_encoder_func=current_le_m3)
        if model_3 and le_precip_trained_for_m3:
            try: model_3.save_model(model_3_path); joblib.dump(le_precip_trained_for_m3, le_3_path); print(f"    Model 3 i LE zapisane.")
            except Exception as e: print(f"    BŁĄD zapisu M3/LE: {e}")
    else: print("    Brak danych treningowych dla M3.")
elif (model_3 is None or le_precip_trained_for_m3 is None) and USE_DATABASE_INPUT: print("  KRYTYCZNY BŁĄD: Model 3 lub LE nie został wczytany."); exit()

# Model 4
model_4_path = os.path.join(MODEL_SAVE_DIR, "model_M4.json"); model_4 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_4_path):
    print(f"  Wczytywanie Modelu 4 z: {model_4_path}")
    try: model_4 = xgb.XGBClassifier(); model_4.load_model(model_4_path); print("    Model 4 wczytany.")
    except Exception as e: print(f"    BŁĄD wczytywania M4: {e}."); model_4 = None
if (model_4 is None or FORCE_RETRAIN) and not USE_DATABASE_INPUT:
    print(f"  {'Wymuszono' if FORCE_RETRAIN else 'Rozpoczynam'} trening Modelu 4...")
    train_df_m4_subset = train_df[train_df['weather_category'].isin(['Clear/Fair', 'Cloudy/Overcast'])].copy()
    y_train_m4 = (train_df_m4_subset['weather_category'] == 'Cloudy/Overcast').astype(int)
    model_4, _ = train_xgboost_model(train_df_m4_subset, y_train_m4, feature_lists_final['M4'], "Model 4", 'binary:logistic', use_smote_func=USE_SMOTE)
    if model_4: 
        try: model_4.save_model(model_4_path); print(f"    Model 4 zapisany: {model_4_path}")
        except Exception as e: print(f"    BŁĄD zapisu M4: {e}")
elif model_4 is None and USE_DATABASE_INPUT: print("  KRYTYCZNY BŁĄD: Model 4 nie został wczytany."); exit()


# --- Etap 6: Predykcja Hierarchiczna ---
print("\n--- Etap 6: Predykcja Hierarchiczna ---")
models_dict = {'M1': model_1, 'M2': model_2, 'M3': model_3, 'M4': model_4}
models_all_available = all(m is not None for m in models_dict.values())
if not models_all_available: print("  KRYTYCZNY BŁĄD: Nie wszystkie modele są dostępne. Przerywam predykcję."); exit()
if le_precip_trained_for_m3 is None and model_3 is not None: # Dodatkowe sprawdzenie dla M3
    print("  KRYTYCZNY BŁĄD: Model M3 jest dostępny, ale jego LabelEncoder (le_precip_trained_for_m3) nie. Przerywam."); exit()


# Wybór danych do predykcji
if USE_DATABASE_INPUT:
    if prediction_data_df.empty: print("  Brak danych do predykcji z bazy. Przerywam."); exit()
    # Sprawdzenie czy wszystkie potrzebne cechy są w prediction_data_df
    missing_cols_in_pred_data = [col for col in all_unique_features_needed_by_models if col not in prediction_data_df.columns]
    if missing_cols_in_pred_data:
        print(f"  BŁĄD KRYTYCZNY: Brakuje następujących cech w danych do predykcji: {missing_cols_in_pred_data}"); exit()
    X_predict_source_df = prediction_data_df[all_unique_features_needed_by_models].copy()
    print(f"  Predykcja na {len(X_predict_source_df)} próbkach z bazy danych.")
elif not test_df.empty: # Tryb Meteostat, użyj test_df
    missing_cols_in_test_df = [col for col in all_unique_features_needed_by_models if col not in test_df.columns]
    if missing_cols_in_test_df:
        print(f"  BŁĄD KRYTYCZNY: Brakuje następujących cech w test_df (Meteostat): {missing_cols_in_test_df}"); exit()
    X_predict_source_df = test_df[all_unique_features_needed_by_models].copy()
    print(f"  Predykcja/Ewaluacja na {len(X_predict_source_df)} próbkach testowych z Meteostat (rok {test_year}).")
else:
    print("  BŁĄD: Brak danych do predykcji (ani z bazy, ani test_df z Meteostat)."); exit()

final_predictions_series = pd.Series(index=X_predict_source_df.index, dtype=object, name='predicted_category')

# Etap 1: Predykcja Opady/Brak (M1)
print("  Etap 1 (M1): Opady vs Brak...")
if feature_lists_final['M1']:
    pred_m1_binary = model_1.predict(X_predict_source_df[feature_lists_final['M1']])
    indices_pred_precip = X_predict_source_df.index[pred_m1_binary == 1]
    indices_pred_no_precip = X_predict_source_df.index[pred_m1_binary == 0]
    print(f"    Przewidziano Opady: {len(indices_pred_precip)}, Brak Opadów: {len(indices_pred_no_precip)}.")
else: # Powinno być obsłużone przez weryfikację cech, ale na wszelki wypadek
    print("    BŁĄD: Brak cech dla Modelu 1. Wszystkie próbki traktowane jako 'Brak Opadów'.")
    indices_pred_precip = pd.Index([])
    indices_pred_no_precip = X_predict_source_df.index

# Etap 2: Predykcja Brak Opadów (M2 - Mgła vs Reszta)
print("  Etap 2 (M2): Mgła vs Inne Bez Opadów...")
indices_pred_fog = pd.Index([])
indices_pred_other_no_precip = pd.Index([]) # Te pójdą do M4
if not indices_pred_no_precip.empty:
    if feature_lists_final['M2']:
        X_m2_subset = X_predict_source_df.loc[indices_pred_no_precip, feature_lists_final['M2']]
        pred_m2_binary = model_2.predict(X_m2_subset)
        indices_pred_fog = X_m2_subset.index[pred_m2_binary == 1]
        indices_pred_other_no_precip = X_m2_subset.index[pred_m2_binary == 0]
        final_predictions_series.loc[indices_pred_fog] = 'Fog'
        print(f"    Przewidziano Mgła: {len(indices_pred_fog)}.")
        print(f"    Pozostałe próbki bez opadów (nie-mgła) do M4: {len(indices_pred_other_no_precip)}.")
    else:
        print("    BŁĄD: Brak cech dla Modelu 2. Wszystkie próbki 'Brak Opadów' idą do M4.")
        indices_pred_other_no_precip = indices_pred_no_precip # Wszystko co było "no_precip" idzie do M4
else:
    print("    Brak próbek 'Brak Opadów' z Etapu 1 dla M2.")


# Etap 3: Predykcja Opady (M3 - Typy Opadów)
print("  Etap 3 (M3): Typy Opadów...")
if not indices_pred_precip.empty:
    if feature_lists_final['M3'] and le_precip_trained_for_m3:
        X_m3_subset = X_predict_source_df.loc[indices_pred_precip, feature_lists_final['M3']]
        pred_m3_numeric = model_3.predict(X_m3_subset)
        try:
            pred_m3_labels = le_precip_trained_for_m3.inverse_transform(pred_m3_numeric)
            final_predictions_series.loc[indices_pred_precip] = pred_m3_labels
            print(f"    Przypisano typy dla {len(indices_pred_precip)} próbek opadowych.")
            # print(f"      Rozkład przewidzianych typów opadów: {pd.Series(pred_m3_labels).value_counts().to_dict()}")
        except ValueError as e_le:
            print(f"    BŁĄD przy odwracaniu transformacji LabelEncoder dla M3: {e_le}. Próbki opadowe bez klasyfikacji.")
    else:
        missing_reason = []
        if not feature_lists_final['M3']: missing_reason.append("brak cech dla M3")
        if not le_precip_trained_for_m3: missing_reason.append("brak LabelEncodera dla M3")
        print(f"    BŁĄD: Nie można sklasyfikować typów opadów ({', '.join(missing_reason)}). Próbki 'Opady' pozostaną bez szczegółowej klasyfikacji.")
else:
    print("    Brak próbek 'Opady' z Etapu 1 dla M3.")

# Etap 4: Predykcja Inne Bez Opadów (M4 - Clear/Fair vs Cloudy/Overcast)
print("  Etap 4 (M4): Clear/Fair vs Cloudy/Overcast...")
if not indices_pred_other_no_precip.empty:
    if feature_lists_final['M4']:
        X_m4_subset = X_predict_source_df.loc[indices_pred_other_no_precip, feature_lists_final['M4']]
        pred_m4_binary = model_4.predict(X_m4_subset)
        # Model M4: 0 to 'Clear/Fair', 1 to 'Cloudy/Overcast'
        final_predictions_series.loc[X_m4_subset.index[pred_m4_binary == 0]] = 'Clear/Fair'
        final_predictions_series.loc[X_m4_subset.index[pred_m4_binary == 1]] = 'Cloudy/Overcast'
        print(f"    Przypisano 'Clear/Fair' lub 'Cloudy/Overcast' dla {len(indices_pred_other_no_precip)} próbek.")
    else:
        print("    BŁĄD: Brak cech dla Modelu 4. Próbki 'Inne Bez Opadów' pozostaną bez klasyfikacji Clear/Cloudy.")
else:
    print("    Brak próbek 'Inne Bez Opadów' z Etapu 2 dla M4.")

# Podsumowanie predykcji
missing_final_preds = final_predictions_series.isnull().sum()
if missing_final_preds > 0:
    print(f"\n  OSTRZEŻENIE: {missing_final_preds} próbek nie otrzymało finalnej predykcji!")
    # Można wypełnić domyślną wartością lub zostawić NaN
    # final_predictions_series.fillna("Unknown_Pred_Error", inplace=True)

# --- Etap 7: Wyniki i Ewaluacja (jeśli dotyczy) ---
print("\n--- Etap 7: Wyniki Predykcji / Ewaluacja ---")

# Zapis predykcji do pliku CSV
if not final_predictions_series.empty:
    results_df = pd.DataFrame({
        'timestamp': final_predictions_series.index,
        'predicted_weather_category': final_predictions_series.values
    })
    # Dodaj oryginalne dane wejściowe (zagregowane godzinowe) do wyników dla kontekstu
    if USE_DATABASE_INPUT and not df_hourly_from_db.empty:
        results_df = results_df.join(df_hourly_from_db[['temp', 'rhum', 'pres', 'wspd', 'prcp', 'tsun', 'snow', 'wpgt']], on='timestamp', how='left')
    
    pred_start_str = PREDICTION_START_DATE.strftime('%Y%m%d_%H%M') if USE_DATABASE_INPUT else f"MeteoTest_{test_year}"
    pred_end_str = PREDICTION_END_DATE.strftime('%Y%m%d_%H%M') if USE_DATABASE_INPUT else ""
    
    results_filename = f"predictions_{pred_start_str}{'_' if pred_end_str else ''}{pred_end_str}.csv"
    results_filepath = os.path.join(MODEL_SAVE_DIR, results_filename)
    try:
        results_df.to_csv(results_filepath, index=False, float_format='%.2f')
        print(f"  Zapisano predykcje do pliku: {results_filepath}")
    except Exception as e_csv:
        print(f"  BŁĄD zapisu predykcji do CSV: {e_csv}")
    print("\n  Przykładowe predykcje:")
    print(results_df.head())
else:
    print("  Brak wygenerowanych predykcji do wyświetlenia lub zapisania.")


# Ewaluacja - tylko jeśli nie używamy danych z bazy (czyli tryb Meteostat z test_df)
# i jeśli y_test_actual_str_meteostat jest dostępne
if not USE_DATABASE_INPUT and 'y_test_actual_str_meteostat' in locals() and not y_test_actual_str_meteostat.empty:
    print(f"\n  Rozpoczynam ewaluację dla danych testowych Meteostat (rok {test_year})...")
    
    # Funkcja ewaluacji (skopiowana z Twojego skryptu)
    def evaluate_model_performance(model_obj, X_test_data, y_test_actuals, model_eval_name, label_enc=None):
        # ... (pełna definicja funkcji evaluate_model z Twojego skryptu)
        print(f"\n--- Ewaluacja: {model_eval_name} ---")
        if model_obj is None: print("   Model nie jest dostępny."); return
        if X_test_data.empty: print("   Brak danych X_test."); return
        if y_test_actuals.empty: print("   Brak danych y_test."); return
        try:
            y_pred_proba_eval = model_obj.predict_proba(X_test_data)
            present_labels_true_eval = sorted(y_test_actuals.unique())
            y_pred_str_eval = None; labels_for_cm_eval = []
            if model_obj.objective == 'binary:logistic':
                y_pred_numeric_eval = (y_pred_proba_eval[:, 1] > 0.5).astype(int)
                map_0, map_1 = 'Brak Opadów', 'Opady' # Domyślne dla M1
                if "Model 1" in model_eval_name: pass
                elif "Model 2" in model_eval_name: map_0, map_1 = 'Inne Bez Opadów', 'Mgła'
                elif "Model 4" in model_eval_name: map_0, map_1 = 'Clear/Fair', 'Cloudy/Overcast'
                else:
                    if len(present_labels_true_eval) == 2: map_0, map_1 = present_labels_true_eval[0], present_labels_true_eval[1]
                    else: print("   BŁĄD: Nie można ustalić mapowania binarnego."); return
                y_pred_str_eval = np.where(y_pred_numeric_eval == 1, map_1, map_0)
                labels_for_cm_eval = [map_0, map_1]
            elif model_obj.objective == 'multi:softmax':
                if label_enc is not None:
                    y_pred_numeric_eval = np.argmax(y_pred_proba_eval, axis=1)
                    try: y_pred_str_eval = label_enc.inverse_transform(y_pred_numeric_eval); labels_for_cm_eval = sorted(list(label_enc.classes_))
                    except ValueError as e_le_inv: print(f"   BŁĄD dekodowania M3: {e_le_inv}"); return
                else: print("   BŁĄD: Brak label_encodera dla M3."); return
            else: print(f"   BŁĄD: Nieobsługiwany cel: {model_obj.objective}"); return
            if y_pred_str_eval is None: print("   BŁĄD: Nie wygenerowano predykcji stringów."); return
            
            all_present_labels_eval = sorted(list(set(y_test_actuals.unique()) | set(np.unique(y_pred_str_eval))))
            final_labels_order_eval = [lbl for lbl in labels_for_cm_eval if lbl in all_present_labels_eval] if labels_for_cm_eval else all_present_labels_eval
            missing_in_order_eval = [lbl for lbl in all_present_labels_eval if lbl not in final_labels_order_eval]
            final_labels_order_eval.extend(missing_in_order_eval)

            accuracy_val = accuracy_score(y_test_actuals, y_pred_str_eval)
            print(f"   Dokładność: {accuracy_val:.4f}")
            print("   Raport Klasyfikacji:"); print(classification_report(y_test_actuals, y_pred_str_eval, labels=final_labels_order_eval, zero_division=0, digits=3))
            cm_eval = confusion_matrix(y_test_actuals, y_pred_str_eval, labels=final_labels_order_eval)
            cm_df_eval = pd.DataFrame(cm_eval, index=final_labels_order_eval, columns=final_labels_order_eval)
            print("   Macierz Pomyłek:"); print(cm_df_eval)
            if VIZ_AVAILABLE:
                try:
                    plt.figure(figsize=(max(6, len(final_labels_order_eval)*1.5), max(5, len(final_labels_order_eval)*1.2)))
                    sns.heatmap(cm_df_eval, annot=True, fmt='d', cmap='Blues'); plt.title(f'Macierz Pomyłek - {model_eval_name}\n(Acc: {accuracy_val:.3f})'); plt.xlabel('Przewidywana'); plt.ylabel('Rzeczywista'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
                    plt.savefig(os.path.join(MODEL_SAVE_DIR, f"cm_{model_eval_name.replace(':', '').replace('/', '').replace(' ', '_')}_eval.png"), dpi=150, bbox_inches='tight'); plt.close()
                except Exception as plot_e: print(f"   OSTRZ.: Błąd wizualizacji CM: {plot_e}")
            print("\n   Top 10 Cech:")
            try:
                importances = model_obj.feature_importances_; feature_names_imp = X_test_data.columns
                fi_df = pd.DataFrame({'feature': feature_names_imp, 'importance': importances}).sort_values(by='importance', ascending=False)
                print(fi_df.head(10).to_string(index=False))
                if VIZ_AVAILABLE:
                    plt.figure(figsize=(10, 6)); sns.barplot(x='importance', y='feature', data=fi_df.head(10), palette='viridis'); plt.title(f'Top 10 Cech - {model_eval_name}'); plt.xlabel('Ważność'); plt.ylabel('Cecha'); plt.tight_layout()
                    plt.savefig(os.path.join(MODEL_SAVE_DIR, f"fi_{model_eval_name.replace(':', '').replace('/', '').replace(' ', '_')}_eval.png"), dpi=150, bbox_inches='tight'); plt.close()
            except Exception as fi_e: print(f"   BŁĄD przetwarzania ważności cech: {fi_e}")
        except Exception as e_eval_main: print(f"   KRYTYCZNY BŁĄD ewaluacji {model_eval_name}: {e_eval_main}"); traceback.print_exc()
        print(f"--- Koniec Ewaluacji: {model_eval_name} ---")

    # Ewaluacja ogólna dla danych Meteostat
    common_indices_eval = y_test_actual_str_meteostat.index.intersection(final_predictions_series.index)
    if not common_indices_eval.empty:
        y_test_eval = y_test_actual_str_meteostat.loc[common_indices_eval]
        y_pred_eval = final_predictions_series.loc[common_indices_eval]
        
        if not y_pred_eval.empty:
            print(f"\n--- === Ewaluacja Końcowa Modelu Hierarchicznego (Meteostat, rok {test_year}, {len(y_test_eval)} próbek) === ---")
            overall_accuracy_eval = accuracy_score(y_test_eval, y_pred_eval)
            print(f"   Dokładność Ogólna: {overall_accuracy_eval:.4f}")
            present_labels_overall_eval = sorted(list(set(y_test_eval.unique()) | set(y_pred_eval.unique())))
            labels_for_cm_overall_eval = sorted(list(set(all_categories_user) | set(present_labels_overall_eval)))
            print(classification_report(y_test_eval, y_pred_eval, labels=present_labels_overall_eval, zero_division=0, digits=3))
            cm_overall_eval = confusion_matrix(y_test_eval, y_pred_eval, labels=labels_for_cm_overall_eval)
            cm_overall_df_eval = pd.DataFrame(cm_overall_eval, index=labels_for_cm_overall_eval, columns=labels_for_cm_overall_eval)
            cm_overall_df_filtered_eval = cm_overall_df_eval.loc[present_labels_overall_eval, present_labels_overall_eval]
            cm_overall_df_filtered_eval = cm_overall_df_filtered_eval.loc[(cm_overall_df_filtered_eval.sum(axis=1) != 0), (cm_overall_df_filtered_eval.sum(axis=0) != 0)]
            print("\n   Macierz Pomyłek Ogólna (Meteostat):"); print(cm_overall_df_filtered_eval)
            if VIZ_AVAILABLE and not cm_overall_df_filtered_eval.empty:
                plt.figure(figsize=(max(8, len(cm_overall_df_filtered_eval.columns)*1.2), max(6, len(cm_overall_df_filtered_eval.index)*1)))
                sns.heatmap(cm_overall_df_filtered_eval, annot=True, fmt='d', cmap='YlGnBu')
                plt.title(f'Macierz Pomyłek Ogólna - Meteostat Test\nRok: {test_year} (Acc: {overall_accuracy_eval:.3f})')
                plt.xlabel('Przewidywana Kategoria'); plt.ylabel('Rzeczywista Kategoria'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
                plt.savefig(os.path.join(MODEL_SAVE_DIR, f"cm_OGOLNA_Meteostat_rok{test_year}.png"), dpi=150, bbox_inches='tight'); plt.close()
        else:
            print("  Brak predykcji do ewaluacji dla danych Meteostat.")
    else:
        print("  Brak wspólnych indeksów do ewaluacji dla danych Meteostat.")

    # Ewaluacja indywidualnych modeli dla danych Meteostat
    print("\n--- === Ewaluacja Modeli Składowych (Meteostat) === ---")
    if model_1 and feature_lists_final['M1']:
        y_test_m1_actual = y_test_actual_str_meteostat.apply(lambda x: 'Opady' if x in precip_categories_user else 'Brak Opadów')
        X_test_m1_data = test_df[feature_lists_final['M1']].copy()
        common_m1_idx = y_test_m1_actual.index.intersection(X_test_m1_data.index)
        if not common_m1_idx.empty: evaluate_model_performance(model_1, X_test_m1_data.loc[common_m1_idx], y_test_m1_actual.loc[common_m1_idx], "M1 (Meteostat)")
    
    if model_2 and feature_lists_final['M2']:
        test_df_m2_actual = test_df[test_df['weather_category'].isin(no_precip_categories_user)]
        if not test_df_m2_actual.empty:
            y_test_m2_actual = test_df_m2_actual['weather_category'].apply(lambda x: 'Mgła' if x == 'Fog' else 'Inne Bez Opadów')
            X_test_m2_data = test_df_m2_actual[feature_lists_final['M2']].copy()
            common_m2_idx = y_test_m2_actual.index.intersection(X_test_m2_data.index)
            if not common_m2_idx.empty: evaluate_model_performance(model_2, X_test_m2_data.loc[common_m2_idx], y_test_m2_actual.loc[common_m2_idx], "M2 (Meteostat)")

    if model_3 and le_precip_trained_for_m3 and feature_lists_final['M3']:
        test_df_m3_actual = test_df[test_df['weather_category'].isin(precip_categories_user)]
        if not test_df_m3_actual.empty:
            y_test_m3_actual = test_df_m3_actual['weather_category']
            X_test_m3_data = test_df_m3_actual[feature_lists_final['M3']].copy()
            common_m3_idx = y_test_m3_actual.index.intersection(X_test_m3_data.index)
            if not common_m3_idx.empty: evaluate_model_performance(model_3, X_test_m3_data.loc[common_m3_idx], y_test_m3_actual.loc[common_m3_idx], "M3 (Meteostat)", label_enc=le_precip_trained_for_m3)

    if model_4 and feature_lists_final['M4']:
        test_df_m4_actual = test_df[test_df['weather_category'].isin(['Clear/Fair', 'Cloudy/Overcast'])]
        if not test_df_m4_actual.empty:
            y_test_m4_actual = test_df_m4_actual['weather_category']
            X_test_m4_data = test_df_m4_actual[feature_lists_final['M4']].copy()
            common_m4_idx = y_test_m4_actual.index.intersection(X_test_m4_data.index)
            if not common_m4_idx.empty: evaluate_model_performance(model_4, X_test_m4_data.loc[common_m4_idx], y_test_m4_actual.loc[common_m4_idx], "M4 (Meteostat)")
else:
    if USE_DATABASE_INPUT:
        print("  Ewaluacja nie jest przeprowadzana w trybie predykcji z bazy (brak rzeczywistych etykiet dla tego okresu).")

print(f"\n--- Skrypt zakończył działanie o {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")