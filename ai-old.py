# -*- coding: utf-8 -*-
from datetime import datetime
import pandas as pd
import numpy as np
import math
import time
import warnings # Do ignorowania niektórych ostrzeżeń
import os ### NOWE ###
import joblib ### NOWE ###

# Ignoruj ostrzeżenia RuntimeWarning i UserWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# Ignoruj FutureWarning z pandas/meteostat - można je później naprawić
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning) # Ignoruj ostrzeżenia o fragmentacji

# --- === Konfiguracja Skryptu === ---
USE_SMOTE = True # Czy używać SMOTE do balansowania klas?
LOAD_MODELS_IF_EXIST = True ### NOWE ### # Czy wczytywać istniejące modele?
FORCE_RETRAIN = False       ### NOWE ### # Czy wymusić ponowny trening, nawet jeśli modele istnieją?
MODEL_SAVE_DIR = "trained_models_wien" ### NOWE ### # Katalog do zapisywania modeli
# --- ========================== ---

# 1. Import bibliotek
try:
    from meteostat import Hourly, Stations
except ImportError:
    print("BŁĄD: Biblioteka Meteostat nie jest zainstalowana.")
    exit()
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import class_weight
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
    from imblearn.over_sampling import SMOTE
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

### NOWE: Utwórz katalog na modele, jeśli nie istnieje ###
if not os.path.exists(MODEL_SAVE_DIR):
    try:
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Utworzono katalog na modele: {MODEL_SAVE_DIR}")
    except OSError as e:
        print(f"BŁĄD: Nie można utworzyć katalogu na modele '{MODEL_SAVE_DIR}': {e}")
        MODEL_SAVE_DIR = "." # Zapisuj w bieżącym katalogu w razie błędu
        print(f"Modele będą zapisywane w katalogu bieżącym.")


# 2. Funkcja do obliczania punktu rosy (bez zmian)
def calculateDewPoint(temperature, humidity):
    if pd.isna(temperature) or pd.isna(humidity) or humidity <= 0 or humidity > 100: return np.nan
    try: a=17.27; b=237.7; gamma=(a * temperature) / (b + temperature) + math.log(humidity / 100.0); dewPoint = (b * gamma) / (a - gamma)
    except (ValueError, OverflowError): return np.nan
    if dewPoint > temperature + 0.5 : return temperature
    if dewPoint < -80: return np.nan
    return dewPoint

# 3. Funkcja Agregacji Coco (user_v2 - 6 klas)
def aggregate_coco_FINAL_user_v2(coco_code):
    """Agreguje kody pogody Meteostat (1-27) do 6 kategorii wg użytkownika."""
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

# --- Ustawienia Danych ---
# *** ZMIENIONE: Tylko Wiedeń ***
target_station_ids = ['11036'] # Wiedeń / Schwechat
station_names = {
    '11036': 'Wien / Schwechat',
}
print(f"Analiza dla stacji: {', '.join([f'{station_names.get(sid, sid)} ({sid})' for sid in target_station_ids])}")

train_start_year = 2018; train_end_year = 2023; test_year = 2024
data_fetch_start_date = datetime(train_start_year, 1, 1)
data_fetch_end_date = datetime(test_year, 12, 31, 23, 59, 59)

print(f"--- Hierarchiczny Model XGBoost v6 (Stacja Wiedeń, Cechy z ANOVA v8) ---")
print(f"   (Kategorie wg user_v2 [6 klas], SMOTE: {'Tak' if USE_SMOTE else 'Nie'})")
print(f"   (Wczytywanie modeli: {'Tak' if LOAD_MODELS_IF_EXIST else 'Nie'}, Wymuszony trening: {'Tak' if FORCE_RETRAIN else 'Nie'})") ### NOWE ###
print(f"Okres pobierania danych: {data_fetch_start_date.strftime('%Y-%m-%d')} - {data_fetch_end_date.strftime('%Y-%m-%d')}")
print(f"Zbiór treningowy: Lata {train_start_year}-{train_end_year}")
print(f"Zbiór testowy: Rok {test_year}")

# --- Pobieranie, Przetwarzanie i Inżynieria Cech (dla Wiednia) ---
print("\n--- Pobieranie, Przetwarzanie Danych i Rozszerzona Inżynieria Cech v2 ---")
full_processing_start_time = time.time()
df_processed = None # Zmienione na df_processed
try:
    # --- Pobieranie Danych (teraz pętla wykona się raz) ---
    print("Pobieranie i wstępne przetwarzanie danych...")
    all_station_data_list = []
    df = None # Inicjalizacja df
    for station_id in target_station_ids:
        print(f"  Pobieranie danych dla stacji: {station_id} ({station_names.get(station_id, '')})...", end="")
        start_fetch_time = time.time(); station_hourly_data = Hourly(station_id, data_fetch_start_date, data_fetch_end_date); station_data = station_hourly_data.fetch(); fetch_duration = time.time() - start_fetch_time
        if station_data.empty: print(f" BRAK DANYCH. Przerywanie."); exit()
        print(f" Pobrano {len(station_data)} rek. w {fetch_duration:.1f}s.")
        required_cols = ['temp', 'rhum', 'coco', 'pres', 'wspd', 'prcp']; optional_cols = ['tsun', 'wpgt', 'snow']
        missing_req = [col for col in required_cols if col not in station_data.columns];
        if missing_req: raise ValueError(f"Brak wymaganych kolumn: {', '.join(missing_req)}")
        for col in optional_cols:
            if col not in station_data.columns: station_data[col] = 0.0
            else: station_data[col] = station_data[col].fillna(0) # Bez inplace
        station_data['prcp'] = station_data['prcp'].fillna(0) # Bez inplace
        station_data.dropna(subset=['temp', 'rhum', 'coco', 'pres', 'wspd'], inplace=True)
        if station_data.empty: raise ValueError(f"Brak danych po usunięciu NaN dla stacji {station_id}.")
        station_data = station_data[station_data['coco'] != 0]; station_data['coco'] = station_data['coco'].astype(int)
        all_station_data_list.append(station_data); print(f"    Przetworzono dane ze stacji {station_id}.")
    df = pd.concat(all_station_data_list) # Połączy listę z jednym elementem
    print(f"  DataFrame zawiera {len(df)} rekordów.")
    print("  Sortowanie danych wg czasu..."); df.sort_index(inplace=True)
    # --- Wspólne przetwarzanie ---
    print("  Agregowanie kategorii (user_v2)..."); df['weather_category'] = df['coco'].apply(aggregate_coco_FINAL_user_v2); df = df[df['weather_category'] != 'Unknown']
    no_precip_categories_user = ['Clear/Fair', 'Cloudy/Overcast', 'Fog']; precip_categories_user = ['Rain', 'Snow/Sleet/Freezing', 'Thunderstorm/Severe']; all_categories_user = sorted(no_precip_categories_user + precip_categories_user)
    print("  Filtrowanie prcp=0 dla opadów..."); initial_rows_before_prcp_filter = len(df); condition_to_remove = (df['prcp'] == 0) & (df['weather_category'].isin(precip_categories_user)); rows_to_remove_count = condition_to_remove.sum()
    if rows_to_remove_count > 0: df = df[~condition_to_remove].copy(); print(f"    Usunięto {rows_to_remove_count} wierszy.")
    if df['weather_category'].nunique() < 2: raise ValueError("Mniej niż 2 kategorie po filtrowaniu.")
    print(f"Wstępne przetwarzanie zakończone. Rekordów: {len(df)}")

    # --- ROZSZERZONA INŻYNIERIA CECH v2 ---
    print("\nRozpoczynanie rozszerzonej inżynierii cech v2...")
    feature_engineering_start_time = time.time(); epsilon = 1e-6
    # (Blok inżynierii cech pozostaje bez zmian - jest długi, więc go pomijam dla zwięzłości odpowiedzi)
    # 1. Cechy Podstawowe i Czasowe
    df['dew_point'] = df.apply(lambda row: calculateDewPoint(row['temp'], row['rhum']), axis=1); df['spread'] = df['temp'] - df['dew_point']; df['hour'] = df.index.hour; df['day_of_year'] = df.index.dayofyear; df['month'] = df.index.month; df['year'] = df.index.year; df['day_of_week'] = df.index.dayofweek; df['week_of_year'] = df.index.isocalendar().week.astype(int); df['quarter'] = df.index.quarter; df['is_weekend'] = (df['day_of_week'] >= 5).astype(int); df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0); df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0); df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/366.0); df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/366.0); df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0); df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0);
    if 'tsun' in df.columns: df['is_daytime_approx'] = (df['tsun'] > 0).astype(int)
    # 2. Różnice Czasowe
    periods_diff = [1, 2, 3, 6, 12, 24]; cols_to_diff = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'prcp', 'tsun', 'wpgt', 'snow']; diff_feature_names = []
    for period in periods_diff:
        for col in cols_to_diff:
            if col in df.columns: diff_col_name = f'{col}_diff_{period}h'; abs_diff_col_name = f'abs_{col}_diff_{period}h'; df[diff_col_name] = df[col].diff(periods=period); df[abs_diff_col_name] = df[diff_col_name].abs(); diff_feature_names.extend([diff_col_name, abs_diff_col_name])
    # 3. Wartości Opóźnione
    periods_lag = [1, 2, 3, 6, 12, 24]; cols_to_lag = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'prcp', 'tsun', 'wpgt', 'snow'] + [f for f in diff_feature_names if '_diff_1h' in f]; lagged_feature_names = []
    for period in periods_lag:
        for col in cols_to_lag:
            if col in df.columns: lag_col_name = f'{col}_lag_{period}h'; df[lag_col_name] = df[col].shift(periods=period); lagged_feature_names.append(lag_col_name)
    # 4. Opóźnione Flagi Kategorii
    lagged_cat_feature_names = []; active_categories = sorted(df['weather_category'].unique())
    for lag in [1, 2, 3, 6]:
        shifted_cat = df['weather_category'].shift(lag)
        for cat in active_categories: safe_cat_name = cat.replace('/', '_').replace(' ', '_').replace('-', '_').lower(); flag_name = f'was_{safe_cat_name}_lag{lag}h'; df[flag_name] = (shifted_cat == cat).astype(int); lagged_cat_feature_names.append(flag_name)
        precip_flag_name = f'was_precip_category_lag{lag}h'; df[precip_flag_name] = shifted_cat.isin(precip_categories_user).astype(int); lagged_cat_feature_names.append(precip_flag_name)
    # 5. Statystyki Kroczące
    window_sizes = [3, 6, 12, 24]; cols_for_rolling = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'prcp', 'tsun', 'wpgt']; rolling_feature_names = []
    for window in window_sizes:
        for col in cols_for_rolling:
            if col in df.columns:
                rolling_window = df[col].rolling(window=window, closed='right', min_periods=max(1, window//2))
                ops = {'mean': rolling_window.mean, 'std': rolling_window.std, 'median': rolling_window.median, 'min': rolling_window.min, 'max': rolling_window.max};
                if col in ['prcp', 'tsun']: ops['sum'] = rolling_window.sum
                for op_name, op_func in ops.items(): feat_name = f'{col}_roll{window}h_{op_name}'; df[feat_name] = op_func(); rolling_feature_names.append(feat_name)
    # 6. Interakcje i Cechy Pochodne
    derived_feature_names = []; base_cols = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd', 'tsun', 'wpgt']
    for i in range(len(base_cols)): # Podstawowe interakcje, potęgi, cechy względne
        for j in range(i, len(base_cols)):
            col1, col2 = base_cols[i], base_cols[j]
            if col1 in df.columns and col2 in df.columns:
                if f'{col1}_x_{col2}' not in df.columns: df[f'{col1}_x_{col2}'] = df[col1] * df[col2]; derived_feature_names.append(f'{col1}_x_{col2}')
                if f'{col1}_div_{col2}_safe' not in df.columns: df[f'{col1}_div_{col2}_safe'] = df[col1] / (df[col2] + epsilon); derived_feature_names.append(f'{col1}_div_{col2}_safe')
                if i != j and f'{col2}_div_{col1}_safe' not in df.columns: df[f'{col2}_div_{col1}_safe'] = df[col2] / (df[col1] + epsilon); derived_feature_names.append(f'{col2}_div_{col1}_safe')
    for col in base_cols + ['prcp']:
        if col in df.columns:
            if f'{col}_pow2' not in df.columns: df[f'{col}_pow2'] = df[col].pow(2); derived_feature_names.append(f'{col}_pow2')
            if col in ['wspd', 'spread', 'prcp'] and f'{col}_pow3' not in df.columns: df[f'{col}_pow3'] = df[col].pow(3); derived_feature_names.append(f'{col}_pow3')
    if all(c in df.columns for c in ['temp_diff_1h', 'wspd']): df['abs_temp_diff_div_wspd_safe'] = df['temp_diff_1h'].abs() / (df['wspd'] + epsilon); derived_feature_names.append('abs_temp_diff_div_wspd_safe')
    if all(c in df.columns for c in ['pres_diff_1h', 'wspd']): df['abs_pres_diff_div_wspd_safe'] = df['pres_diff_1h'].abs() / (df['wspd'] + epsilon); derived_feature_names.append('abs_pres_diff_div_wspd_safe')
    if all(c in df.columns for c in ['temp', 'pres_diff_1h']): df['temp_x_abs_pres_diff_1h'] = df['temp'] * df['pres_diff_1h'].abs(); derived_feature_names.append('temp_x_abs_pres_diff_1h')
    if all(c in df.columns for c in ['rhum', 'spread_diff_1h']): df['rhum_x_abs_spread_diff_1h'] = df['rhum'] * df['spread_diff_1h'].abs(); derived_feature_names.append('rhum_x_abs_spread_diff_1h')
    if all(c in df.columns for c in ['prcp_lag_1h', 'temp']): df['prcp_lag_1h_x_temp'] = df['prcp_lag_1h'] * df['temp']; derived_feature_names.append('prcp_lag_1h_x_temp')
    for window in window_sizes:
        for col in ['temp', 'rhum', 'spread', 'pres', 'wspd']:
            mean_col_name = f'{col}_roll{window}h_mean'; rel_col_name = f'{col}_rel_to_roll{window}h_mean'
            if col in df.columns and mean_col_name in df.columns: df[rel_col_name] = df[col] - df[mean_col_name]; derived_feature_names.append(rel_col_name)
    # 7. Dodatkowe Cechy Matematyczne
    additional_math_features = []; cols_for_adv_math = ['temp', 'rhum', 'dew_point', 'spread', 'pres', 'wspd']
    for window in window_sizes: # Zakres kroczący, Stosunek do Std krocz., Zmienność zmian
        for col in cols_for_adv_math:
            min_col=f'{col}_roll{window}h_min'; max_col=f'{col}_roll{window}h_max'; range_col=f'{col}_roll{window}h_range'; std_col=f'{col}_roll{window}h_std'; ratio_std_col=f'{col}_div_roll{window}h_std_safe'; diff_1h_col=f'{col}_diff_1h'; volatility_col=f'{diff_1h_col}_roll{window}h_std'
            if min_col in df.columns and max_col in df.columns: df[range_col] = df[max_col] - df[min_col]; additional_math_features.append(range_col)
            if col in df.columns and std_col in df.columns: df[ratio_std_col] = df[col] / (df[std_col] + epsilon); additional_math_features.append(ratio_std_col)
            if diff_1h_col in df.columns: df[volatility_col] = df[diff_1h_col].rolling(window=window, min_periods=max(1, window//2)).std(); additional_math_features.append(volatility_col)
    for col in cols_for_adv_math: # "Przyspieszenie"
        diff_1h_col = f'{col}_diff_1h'; accel_col = f'{diff_1h_col}_diff_1h';
        if diff_1h_col in df.columns: df[accel_col] = df[diff_1h_col].diff(1); additional_math_features.append(accel_col)
    if 'dew_point' in df.columns and 'temp' in df.columns: df['dp_div_temp_safe'] = df['dew_point'] / (df['temp'] + np.sign(df['temp'])*epsilon + epsilon); additional_math_features.append('dp_div_temp_safe')
    if 'spread' in df.columns and 'temp' in df.columns: df['spread_div_temp_safe'] = df['spread'] / (df['temp'] + np.sign(df['temp'])*epsilon + epsilon); additional_math_features.append('spread_div_temp_safe')
    if 'temp' in df.columns and 'hour_sin' in df.columns: df['temp_x_hour_sin'] = df['temp'] * df['hour_sin']; additional_math_features.append('temp_x_hour_sin')
    if 'wspd' in df.columns and 'day_of_year_cos' in df.columns: df['wspd_x_day_of_year_cos'] = df['wspd'] * df['day_of_year_cos']; additional_math_features.append('wspd_x_day_of_year_cos')
    if 'rhum' in df.columns and 'pres_diff_1h' in df.columns: df['rhum_x_pres_diff_1h'] = df['rhum'] * df['pres_diff_1h']; additional_math_features.append('rhum_x_pres_diff_1h')
    # 8. Rozszerzone Flagi Binarne
    thresholds = { # Pełny słownik flag jak w ANOVA v6/v7
        'flag_sunny': ('tsun', '>', 45), 'flag_partly_sunny': ('tsun', '>', 15), 'flag_mostly_dark': ('tsun', '<', 10), 'flag_dark': ('tsun', '<', 1),
        'flag_very_humid': ('rhum', '>', 96), 'flag_humid': ('rhum', '>', 85), 'flag_moderate_humid': ('rhum', '<=', 85), 'flag_dry': ('rhum', '<', 65), 'flag_very_dry': ('rhum', '<', 45),
        'flag_near_saturation': ('spread', '<', 0.7), 'flag_close_saturation': ('spread', '<', 1.5), 'flag_moderate_spread': ('spread', '>', 3.0), 'flag_far_from_saturation': ('spread', '>', 6.0),
        'flag_calm': ('wspd', '<', 2.5), 'flag_light_breeze': ('wspd', '>', 5.0), 'flag_moderate_wind': ('wspd', '>', 8.0), 'flag_windy': ('wspd', '>', 12.0), 'flag_very_windy': ('wspd', '>', 18.0),
        'flag_has_precip': ('prcp', '>', 0), 'flag_light_precip': ('prcp', '>', 0.1), 'flag_moderate_precip': ('prcp', '>', 1.0), 'flag_heavy_precip': ('prcp', '>', 3.0),
        'flag_very_high_pres': ('pres', '>', 1030), 'flag_high_pres': ('pres', '>', 1020), 'flag_moderate_pres': ('pres', '<=', 1020), 'flag_low_pres': ('pres', '<', 1005), 'flag_very_low_pres': ('pres', '<', 990),
        'flag_below_freezing': ('temp', '<=', 0), 'flag_near_freezing_low': ('temp', '<=', 2), 'flag_near_freezing_high': ('temp', '<=', 5), 'flag_cool': ('temp', '<', 10), 'flag_mild': ('temp', '>=', 10), 'flag_warm': ('temp', '>', 18), 'flag_hot': ('temp', '>', 25),
        'flag_dp_below_freezing': ('dew_point', '<=', 0), 'flag_dp_low': ('dew_point', '<', 5), 'flag_dp_high': ('dew_point', '>', 15), 'flag_dp_very_high': ('dew_point', '>', 20),
        'flag_has_gusts': ('wpgt', '>', 0), 'flag_strong_gusts': ('wpgt', '>', 15), 'flag_severe_gusts': ('wpgt', '>', 25),
        'flag_has_snow_cover': ('snow', '>', 0), 'flag_sig_snow_cover': ('snow', '>', 50),
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
        'flag_damp_conditions': (('rhum', '>', 92), ('spread', '<', 1.5)), 'flag_frontal_drizzle_potential': (('flag_pres_falling_sig', '==', 1), ('rhum', '>', 85), ('spread', '<', 3.0)),}
    calculated_flags = set(); flags_to_calculate = list(thresholds.keys()); max_iterations = 5; iteration = 0; threshold_flags_names = []
    # --- Iteracyjne obliczanie flag ---
    print("   Obliczanie flag binarnych...")
    # Tworzenie kopii słownika thresholds, aby można było z niego usuwać
    thresholds_copy = thresholds.copy()
    while flags_to_calculate and iteration < max_iterations:
        newly_calculated = []; remaining_flags = [];
        # Iteruj po kopii listy flags_to_calculate, aby można było modyfikować oryginał
        for flag_name in list(flags_to_calculate):
            # Sprawdź, czy flaga nadal jest w thresholds_copy (nie została usunięta z powodu błędu)
            if flag_name not in thresholds_copy:
                if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name)
                continue

            conditions = thresholds_copy[flag_name]; can_calculate = True; required_features = []
            try:
                is_combined = isinstance(conditions[0], tuple)
                current_required_features = [cond[0] for cond in conditions] if is_combined else [conditions[0]]

                for feature in current_required_features:
                    is_dependency_flag = feature.startswith('flag_')
                    if is_dependency_flag and feature not in calculated_flags:
                        can_calculate = False; break
                    elif not is_dependency_flag and feature not in df.columns:
                        # To jest błąd - cecha bazowa nie istnieje. Zgłoś i usuń flagę.
                        print(f"    BŁĄD KRYTYCZNY: Bazowa cecha '{feature}' dla flagi '{flag_name}' nie istnieje w DataFrame. Pomijam tę flagę.")
                        if flag_name in thresholds_copy: del thresholds_copy[flag_name]
                        if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name)
                        can_calculate = False; break # Przerwij sprawdzanie cech dla tej flagi
                
                if not can_calculate and flag_name in flags_to_calculate and flag_name in thresholds_copy : # Jeśli nie można obliczyć z powodu zależności lub braku cechy
                    remaining_flags.append(flag_name) # Dodaj do pozostałych, jeśli to tylko zależność
                    continue # Przejdź do następnej flagi

                if can_calculate : # Jeśli można obliczyć (wszystkie cechy bazowe i flagi zależne są dostępne)
                    final_condition = pd.Series(True, index=df.index)
                    process_cond = lambda feat, op, th: {'<': df[feat] < th, '>': df[feat] > th, '<=': df[feat] <= th, '>=': df[feat] >= th, '==': df[feat] == th, '!=': df[feat] != th}.get(op, pd.Series(False, index=df.index))
                    
                    if is_combined:
                        for feature, operator, threshold in conditions:
                            if feature not in df.columns: # Podwójne sprawdzenie, na wszelki wypadek
                                raise KeyError(f"Cecha '{feature}' wymagana przez flagę '{flag_name}' nie znaleziona w df podczas budowania warunku.")
                            final_condition &= process_cond(feature, operator, threshold)
                    else:
                        feature, operator, threshold = conditions
                        if feature not in df.columns: # Podwójne sprawdzenie
                            raise KeyError(f"Cecha '{feature}' wymagana przez flagę '{flag_name}' nie znaleziona w df podczas budowania warunku.")
                        final_condition = process_cond(feature, operator, threshold)
                    
                    df[flag_name] = final_condition.astype(int)
                    threshold_flags_names.append(flag_name)
                    calculated_flags.add(flag_name)
                    newly_calculated.append(flag_name)
                    if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name) # Usunięcie z listy do obliczenia
                    if flag_name in thresholds_copy: del thresholds_copy[flag_name] # Usunięcie z kopii słownika
            
            except KeyError as ke: # Błąd jeśli cecha nie istnieje w df
                print(f"    BŁĄD (KeyError) przy obliczaniu flagi '{flag_name}': {ke}. Pomijam tę flagę.")
                if flag_name in thresholds_copy: del thresholds_copy[flag_name]
                if flag_name in flags_to_calculate: flags_to_calculate.remove(flag_name)
                # Nie dodawaj do remaining_flags, bo błąd jest krytyczny dla tej flagi
            except Exception as e:
                print(f"    BŁĄD (Inny) przy obliczaniu flagi '{flag_name}': {e}. Spróbuję później.")
                if flag_name not in remaining_flags and flag_name in thresholds_copy: # Dodaj do pozostałych, jeśli jeszcze nie ma i jest w słowniku
                    remaining_flags.append(flag_name)
                # Nie usuwaj z thresholds_copy, jeśli to błąd przejściowy

        flags_to_calculate = remaining_flags # Aktualizuj listę flag do obliczenia
        iteration += 1
        if not newly_calculated and flags_to_calculate:
            print(f"    OSTRZEŻENIE: W iteracji {iteration} nie udało się obliczyć żadnych nowych flag. Pozostałe flagi: {flags_to_calculate}")
            # Można dodać logikę sprawdzania, dlaczego nie udało się obliczyć
            unresolved_dependencies = {}
            for fname in flags_to_calculate:
                if fname in thresholds_copy:
                    conds = thresholds_copy[fname]
                    is_comb = isinstance(conds[0], tuple)
                    req_feats_current = [c[0] for c in conds] if is_comb else [conds[0]]
                    missing_deps = [rf for rf in req_feats_current if rf.startswith('flag_') and rf not in calculated_flags]
                    if missing_deps:
                        unresolved_dependencies[fname] = missing_deps
            if unresolved_dependencies:
                print(f"    Nierozwiązane zależności dla pozostałych flag: {unresolved_dependencies}")
            break # Przerwij pętlę, jeśli nie ma postępu
    
    if flags_to_calculate:
        print(f"    OSTRZEŻENIE: Nie udało się obliczyć wszystkich flag po {max_iterations} iteracjach. Nieuobliczone flagi: {flags_to_calculate}")

    print(f"   Utworzono {len(threshold_flags_names)} flag binarnych.")
    feature_engineering_duration = time.time() - feature_engineering_start_time
    print(f"--- Zakończono Rozszerzoną Inżynierię Cech v2 ({feature_engineering_duration:.1f} sek) ---")

    # --- FINALNE CZYSZCZENIE NaN ---
    print("\nUsuwanie NaN po pełnej inżynierii cech (finalne)...")
    rows_before_final_dropna = len(df); potential_feature_cols = list(df.select_dtypes(include=np.number).columns); cols_to_exclude_from_dropna = ['coco', 'year']; features_for_dropna = [f for f in potential_feature_cols if f not in cols_to_exclude_from_dropna]
    df.dropna(subset=features_for_dropna, inplace=True); rows_after_processing = len(df)
    print(f"Usunięto {rows_before_final_dropna - rows_after_processing} wierszy z NaN. Ostateczna liczba rekordów: {rows_after_processing}.")
    if rows_after_processing == 0: raise ValueError("Brak danych po pełnym przetworzeniu i usunięciu NaN.")
    df_processed = df.copy()

except Exception as e:
    print(f"KRYTYCZNY BŁĄD podczas przetwarzania danych lub inżynierii cech: {e}"); import traceback; traceback.print_exc(); exit()
if df_processed is None or df_processed.empty: print("KRYTYCZNY BŁĄD: df_processed jest pusty."); exit()

full_processing_duration = time.time() - full_processing_start_time
print(f"--- Całkowity czas przetwarzania i FE: {full_processing_duration:.1f} sek ---")


# --- Definicje List Cech z ANOVA v8 (Wiedeń) ---
print("\n--- Definiowanie list cech na podstawie wyników ANOVA v8 (Wiedeń) ---")
FEATURES_M1 = [ # Model 1: Opady vs Brak Opadów (v8)
    'was_precip_category_lag1h', 'flag_light_precip', 'flag_has_precip', 'flag_gusty_and_precip',
    'was_rain_lag1h', 'flag_windy_and_precip', 'was_precip_category_lag2h', 'was_rain_lag2h',
    'was_precip_category_lag3h', 'was_rain_lag3h', 'prcp_roll6h_median', 'prcp_roll3h_sum',
    'prcp_roll3h_mean', 'prcp_roll6h_mean', 'prcp_roll6h_sum', 'prcp_roll3h_median',
    'prcp_roll3h_min', 'prcp_roll12h_mean', 'prcp_roll12h_sum', 'prcp_roll12h_median',
    'prcp_roll3h_max', 'prcp', 'prcp_roll6h_min', 'flag_moderate_precip',
    'prcp_roll6h_max', 'prcp_lag_1h', 'was_precip_category_lag6h', 'prcp_roll6h_std',
    'prcp_roll3h_std', 'flag_near_freezing_precip'
]
FEATURES_M2 = [ # Model 2: Mgła vs (Clear/Fair + Cloudy/Overcast) (v8)
    'was_fog_lag1h', 'flag_damp_conditions', 'was_fog_lag2h', 'flag_fog_ratio',
    'flag_near_saturation', 'flag_close_saturation', 'was_fog_lag3h', 'flag_very_humid',
    'flag_cold_and_humid', 'flag_moderate_humid', 'flag_humid', 'rhum_x_spread',
    'was_fog_lag6h', 'rhum_pow2', 'rhum_x_rhum', 'flag_moderate_spread',
    'rhum_roll6h_min', 'rhum_div_roll24h_std_safe', 'rhum_roll3h_min', 'rhum_roll12h_min',
    'rhum_x_pres', 'rhum', 'rhum_roll3h_mean', 'rhum_roll24h_min',
    'rhum_div_pres_safe', 'rhum_roll24h_mean', 'rhum_roll6h_mean', 'rhum_roll3h_median',
    'rhum_roll12h_mean', 'rhum_lag_1h'
]
FEATURES_M3 = [ # Model 3: Typy Opadów (v8)
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
FEATURES_M4 = [ # Model 4: Clear/Fair vs Cloudy/Overcast (v8)
    'was_clear_fair_lag1h', 'was_cloudy_overcast_lag1h', 'was_clear_fair_lag2h', 'was_cloudy_overcast_lag2h',
    'was_clear_fair_lag3h', 'was_cloudy_overcast_lag3h', 'was_clear_fair_lag6h', 'was_cloudy_overcast_lag6h',
    'temp_roll12h_range', 'tsun_roll24h_sum', 'tsun_roll24h_mean', 'temp_roll12h_std',
    'temp_roll24h_std', 'temp_roll24h_range', 'tsun_roll12h_sum', 'tsun_roll12h_mean',
    'spread_roll24h_std', 'spread_roll12h_max', 'rhum_roll12h_min', 'rhum_roll24h_min',
    'spread_roll12h_range', 'spread_roll24h_range', 'spread_roll24h_max', 'spread_roll12h_std',
    'tsun_roll12h_max', 'tsun_roll12h_std', 'abs_tsun_diff_12h', 'rhum_roll24h_std',
    'tsun_roll24h_std', 'tsun_roll6h_mean'
]

# Weryfikacja i finalizacja list cech
print("\nWeryfikacja dostępności cech ANOVA v8 w przetworzonym DataFrame...")
feature_lists_final = {}
all_features_unpacked = []
for name, features in [("M1", FEATURES_M1), ("M2", FEATURES_M2), ("M3", FEATURES_M3), ("M4", FEATURES_M4)]:
    available_features = [f for f in features if f in df_processed.columns]
    missing = sorted(list(set(features) - set(available_features)))
    # ### ZMIENIONE: Sprawdzanie czy cechy są flagami, które mogły nie zostać utworzone ###
    non_flag_missing = [m for m in missing if not m.startswith('flag_')]
    flag_missing_but_not_in_threshold_flags = [m for m in missing if m.startswith('flag_') and m not in threshold_flags_names]

    if non_flag_missing:
         print(f"   KRYTYCZNE OSTRZEŻENIE ({name}): Nie znaleziono NIE-FLAGOWYCH cech: {', '.join(non_flag_missing)}.")
    if flag_missing_but_not_in_threshold_flags:
         print(f"   OSTRZEŻENIE ({name}): Flagi nie znalezione LUB nie utworzone: {', '.join(flag_missing_but_not_in_threshold_flags)}.")
    
    # Informacja o wszystkich brakujących, jeśli są
    if missing:
        # Usuń brakujące cechy z listy `features`, aby nie powodowały błędu później
        print(f"   ({name}): Usunięto {len(missing)} brakujących/nieutworzonych cech z listy dla tego modelu.")
        current_features_for_model = [f for f in features if f in available_features]
    else:
        print(f"   ({name}): Wszystkie {len(features)} cechy są dostępne.")
        current_features_for_model = features # Użyj oryginalnej listy, jeśli wszystko jest OK

    if not current_features_for_model: # Użyj zaktualizowanej listy
        print(f"   KRYTYCZNY BŁĄD ({name}): Brak dostępnych cech po weryfikacji! Model nie może być trenowany.")
        # Można zdecydować o przerwaniu skryptu lub pominięciu tego modelu
        # Na razie zapiszemy pustą listę, co spowoduje błąd przy próbie treningu
        feature_lists_final[name] = []
        # exit() # Można odkomentować, aby przerwać
    else:
        feature_lists_final[name] = current_features_for_model # Zapisz przefiltrowaną lub oryginalną listę
        all_features_unpacked.extend(current_features_for_model) # Dodaj tylko dostępne cechy

all_unique_features_needed = sorted(list(set(all_features_unpacked)))
print(f"Łącznie unikalnych cech potrzebnych przez wszystkie modele (po weryfikacji): {len(all_unique_features_needed)}")


# --- Przygotowanie Danych Treningowych i Testowych ---
print("\n--- Przygotowanie danych Treningowych i Testowych ---")
train_df = df_processed[(df_processed['year'] >= train_start_year) & (df_processed['year'] <= train_end_year)].copy()
test_df = df_processed[df_processed['year'] == test_year].copy()
print(f"Liczba próbek treningowych (przed SMOTE): {len(train_df)}")
print(f"Liczba próbek testowych: {len(test_df)}")
if train_df.empty or test_df.empty: print("BŁĄD: Zbiór treningowy lub testowy jest pusty."); exit()
y_test_actual_str = test_df['weather_category']

# --- Funkcja Pomocnicza do Trenowania i SMOTE ---
def train_xgboost_model(X_train, y_train, features, model_name, objective, num_class=None, use_smote=False, label_encoder=None):
    print(f"\n--- Trenowanie {model_name} ---")
    if not features: ### NOWE: Sprawdzenie, czy lista cech nie jest pusta
        print(f"   BŁĄD: Brak cech do trenowania dla modelu {model_name}.")
        return None, None
    X_train_model = X_train[features]
    y_train_model = y_train
    if X_train_model.empty or len(y_train_model) == 0: print(f"   BŁĄD: Brak danych dla {model_name}."); return None, None
    
    # Sprawdzenie typów danych i konwersja jeśli konieczne
    for col in X_train_model.columns:
        if X_train_model[col].dtype == 'object':
            try:
                X_train_model[col] = pd.to_numeric(X_train_model[col])
            except ValueError:
                print(f"OSTRZEŻENIE: Nie udało się przekonwertować kolumny {col} na typ numeryczny. Może to być problem.")
        # XGBoost nie lubi bool, preferuje int
        if X_train_model[col].dtype == 'bool':
            X_train_model[col] = X_train_model[col].astype(int)


    print(f"   Rozkład klas przed SMOTE: {np.bincount(y_train_model)}")
    X_train_resampled, y_train_resampled = X_train_model, y_train_model; scale_pos_weight_val = 1
    if use_smote and SMOTE_AVAILABLE:
        unique_classes, counts = np.unique(y_train_model, return_counts=True); min_class_count = counts.min() if len(counts)>0 else 0
        if len(unique_classes) > 1 and min_class_count >= 5: # Zmieniono z >5 na >=5 dla k_neighbors
            k_neighbors_smote = min(5, min_class_count - 1)
            if k_neighbors_smote < 1: k_neighbors_smote = 1 # SMOTE wymaga k_neighbors >= 1
            print(f"   Stosowanie SMOTE (min klasa: {min_class_count}, k_neighbors={k_neighbors_smote})...")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            try: X_train_resampled, y_train_resampled = smote.fit_resample(X_train_model, y_train_model); print(f"   Rozkład klas PO SMOTE: {np.bincount(y_train_resampled)}")
            except ValueError as e: print(f"   OSTRZ.: Błąd SMOTE: {e}. Używam oryg. danych.")
        elif len(unique_classes) > 1:
            print(f"   OSTRZ.: Za mało próbek ({min_class_count}) dla SMOTE (wymagane min. 5 i k_neighbors >=1).")
            if len(unique_classes) == 2 and counts[0] > 0 and counts[1] > 0 : # Oblicz scale_pos_weight jeśli binarny
                 scale_pos_weight_val = counts[0] / counts[1]; print(f"   Używam scale_pos_weight = {scale_pos_weight_val:.2f}")
        elif len(unique_classes) <=1:
             print(f"   OSTRZ.: Tylko jedna klasa ({unique_classes}) w danych treningowych. SMOTE nie zostanie zastosowane.")

    elif not use_smote and objective == 'binary:logistic': # Zmienione z 'multi:softmax' na 'binary:logistic'
        counts = np.bincount(y_train_model);
        if len(counts) == 2 and counts[0] > 0 and counts[1] > 0: scale_pos_weight_val = counts[0] / counts[1]; print(f"   SMOTE wyłączone. Używam scale_pos_weight = {scale_pos_weight_val:.2f}")
    
    xgb_params = {'objective': objective, 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.7, 'colsample_bytree': 0.7, 'eval_metric': 'logloss' if 'binary' in objective else 'mlogloss', 'random_state': 42, 'n_jobs': -1}
    if objective == 'binary:logistic' and scale_pos_weight_val != 1: xgb_params['scale_pos_weight'] = scale_pos_weight_val
    if num_class: xgb_params['num_class'] = num_class
    
    model = xgb.XGBClassifier(**xgb_params); print(f"   Trenowanie XGBoost..."); start_time = time.time()
    try: model.fit(X_train_resampled, y_train_resampled); print(f"   Trening zakończony w {time.time() - start_time:.1f} sek."); return model, label_encoder
    except Exception as e: print(f"   BŁĄD treningu: {e}"); return None, None

# --- Funkcja do Ewaluacji Modelu ---
def evaluate_model(model, X_test, y_test_str, model_name, label_encoder=None):
    # (Funkcja ewaluacji pozostaje bez zmian - jest długa, więc ją pomijam dla zwięzłości odpowiedzi)
    print(f"\n--- Ewaluacja: {model_name} ---")
    if model is None: print("   Model nie został wytrenowany."); return
    if X_test.empty: print("   Brak danych testowych (X_test)."); return
    if y_test_str.empty: print("   Brak danych testowych (y_test_str)."); return

    try:
        y_pred_proba = model.predict_proba(X_test)
        present_labels_true = sorted(y_test_str.unique())
        y_pred_str = None; labels_for_confusion_matrix = []

        if model.objective == 'binary:logistic':
            y_pred_numeric = (y_pred_proba[:, 1] > 0.5).astype(int)
            map_0, map_1 = 'Brak Opadów', 'Opady' # Domyślne dla M1
            if "Model 1" in model_name: pass # Już ustawione
            elif "Model 2" in model_name: map_0, map_1 = 'Inne Bez Opadów', 'Mgła'
            elif "Model 4" in model_name: map_0, map_1 = 'Clear/Fair', 'Cloudy/Overcast'
            else: # Fallback
                if len(present_labels_true) == 2: map_0, map_1 = present_labels_true[0], present_labels_true[1]; print(f"   OSTRZEŻENIE: Używam domyślnego mapowania binarnego: 0->{map_0}, 1->{map_1}")
                else: print("   BŁĄD: Nie można ustalić mapowania dla modelu binarnego."); return
            y_pred_str = np.where(y_pred_numeric == 1, map_1, map_0)
            labels_for_confusion_matrix = [map_0, map_1]
        elif model.objective == 'multi:softmax':
            if label_encoder is not None:
                y_pred_numeric = np.argmax(y_pred_proba, axis=1)
                try: y_pred_str = label_encoder.inverse_transform(y_pred_numeric); labels_for_confusion_matrix = sorted(list(label_encoder.classes_))
                except ValueError as e: print(f"   BŁĄD: Nie udało się zdekodować predykcji M3: {e}"); return
            else: print("   BŁĄD: Brak label_encodera dla M3."); return
        else: print(f"   BŁĄD: Nieobsługiwany cel modelu: {model.objective}"); return

        if y_pred_str is None: print("   BŁĄD: Nie udało się wygenerować predykcji stringów."); return
        
        all_present_labels = sorted(list(set(y_test_str.unique()) | set(np.unique(y_pred_str))))
        final_labels_order = [lbl for lbl in labels_for_confusion_matrix if lbl in all_present_labels] if labels_for_confusion_matrix else all_present_labels
        missing_in_order = [lbl for lbl in all_present_labels if lbl not in final_labels_order]
        final_labels_order.extend(missing_in_order)

        accuracy = accuracy_score(y_test_str, y_pred_str)
        print(f"   Dokładność (Accuracy): {accuracy:.4f}")
        print("   Raport Klasyfikacji:"); print(classification_report(y_test_str, y_pred_str, labels=final_labels_order, zero_division=0, digits=3))
        print("   Macierz Pomyłek:"); cm = confusion_matrix(y_test_str, y_pred_str, labels=final_labels_order); cm_df = pd.DataFrame(cm, index=final_labels_order, columns=final_labels_order); print(cm_df)

        if VIZ_AVAILABLE:
            try:
                plt.figure(figsize=(max(6, len(final_labels_order)*1.5), max(5, len(final_labels_order)*1.2)))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues'); plt.title(f'Macierz Pomyłek - {model_name}\n(Accuracy: {accuracy:.3f})'); plt.xlabel('Przewidywana'); plt.ylabel('Rzeczywista'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
                nazwa_pliku_cm = os.path.join(MODEL_SAVE_DIR, f"macierz_pomylek_{model_name.replace(':', '').replace('/', '').replace(' ', '_')}.png") ### ZMIENIONE: Zapis do katalogu ###
                plt.savefig(nazwa_pliku_cm, dpi=150, bbox_inches='tight'); print(f"   Zapisano macierz pomyłek: {nazwa_pliku_cm}"); plt.show()
            except Exception as plot_e: print(f"   OSTRZ.: Błąd wizualizacji/zapisu macierzy pomyłek: {plot_e}")
        
        print("\n   Top 10 Najważniejszych Cech:")
        try:
            importances = model.feature_importances_; feature_names = X_test.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
            print(feature_importance_df.head(10).to_string(index=False))
            if VIZ_AVAILABLE:
                plt.figure(figsize=(10, 6)); sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10), palette='viridis'); plt.title(f'Top 10 Cech - {model_name}'); plt.xlabel('Ważność'); plt.ylabel('Cecha'); plt.tight_layout()
                nazwa_pliku_imp = os.path.join(MODEL_SAVE_DIR, f"waznosc_cech_{model_name.replace(':', '').replace('/', '').replace(' ', '_')}.png") ### ZMIENIONE: Zapis do katalogu ###
                plt.savefig(nazwa_pliku_imp, dpi=150, bbox_inches='tight'); print(f"   Zapisano ważność cech: {nazwa_pliku_imp}"); plt.show()
        except AttributeError: print("   Nie można pobrać ważności cech.")
        except Exception as fi_e: print(f"   BŁĄD przetwarzania/wizualizacji ważności cech: {fi_e}")
    except Exception as e: print(f"   KRYTYCZNY BŁĄD ewaluacji {model_name}: {e}"); import traceback; traceback.print_exc()
    print(f"--- Koniec Ewaluacji: {model_name} ---")


# --- Trening Modeli Hierarchicznych (Używając NOWYCH list cech ANOVA v8) ---
print("\n--- Trening/Wczytywanie Modeli Składowych (Cechy z ANOVA v8 - Wiedeń) ---") ### ZMIENIONE ###

# --- Model 1: Opady vs Brak Opadów ---
model_1_path = os.path.join(MODEL_SAVE_DIR, "model_M1.json")
model_1 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_1_path):
    print(f"\n--- Wczytywanie Modelu 1 z pliku: {model_1_path} ---")
    try:
        model_1 = xgb.XGBClassifier()
        model_1.load_model(model_1_path)
        print("   Model 1 wczytany pomyślnie.")
    except Exception as e:
        print(f"   BŁĄD podczas wczytywania Modelu 1: {e}. Model zostanie wytrenowany od nowa.")
        model_1 = None
if model_1 is None or FORCE_RETRAIN:
    if FORCE_RETRAIN and model_1 is not None: print("   Wymuszono ponowny trening Modelu 1.")
    y_train_m1 = train_df['weather_category'].isin(precip_categories_user).astype(int)
    model_1, _ = train_xgboost_model(train_df, y_train_m1, feature_lists_final['M1'],
                                     "Model 1: Opady vs Brak Opadów", 'binary:logistic', use_smote=USE_SMOTE)
    if model_1:
        try:
            model_1.save_model(model_1_path)
            print(f"   Model 1 zapisany do: {model_1_path}")
        except Exception as e:
            print(f"   BŁĄD podczas zapisywania Modelu 1: {e}")

# --- Model 2: Mgła vs (Clear/Fair + Cloudy/Overcast) ---
model_2_path = os.path.join(MODEL_SAVE_DIR, "model_M2.json")
model_2 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_2_path):
    print(f"\n--- Wczytywanie Modelu 2 z pliku: {model_2_path} ---")
    try:
        model_2 = xgb.XGBClassifier()
        model_2.load_model(model_2_path)
        print("   Model 2 wczytany pomyślnie.")
    except Exception as e:
        print(f"   BŁĄD podczas wczytywania Modelu 2: {e}. Model zostanie wytrenowany od nowa.")
        model_2 = None
if model_2 is None or FORCE_RETRAIN:
    if FORCE_RETRAIN and model_2 is not None: print("   Wymuszono ponowny trening Modelu 2.")
    train_df_m2_subset = train_df[train_df['weather_category'].isin(no_precip_categories_user)].copy()
    y_train_m2 = (train_df_m2_subset['weather_category'] == 'Fog').astype(int)
    model_2, _ = train_xgboost_model(train_df_m2_subset, y_train_m2, feature_lists_final['M2'],
                                     "Model 2: Mgła vs Inne Bez Opadów", 'binary:logistic', use_smote=USE_SMOTE)
    if model_2:
        try:
            model_2.save_model(model_2_path)
            print(f"   Model 2 zapisany do: {model_2_path}")
        except Exception as e:
            print(f"   BŁĄD podczas zapisywania Modelu 2: {e}")

# --- Model 3: Typy Opadów ---
model_3_path = os.path.join(MODEL_SAVE_DIR, "model_M3.json")
le_3_path = os.path.join(MODEL_SAVE_DIR, "le_M3.pkl")
model_3 = None
le_precip_trained_for_m3 = None # Dedykowana zmienna dla LabelEncodera M3

if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_3_path) and os.path.exists(le_3_path):
    print(f"\n--- Wczytywanie Modelu 3 i LabelEncodera z plików ---")
    try:
        model_3 = xgb.XGBClassifier()
        model_3.load_model(model_3_path)
        le_precip_trained_for_m3 = joblib.load(le_3_path)
        print(f"   Model 3 wczytany z: {model_3_path}")
        print(f"   LabelEncoder dla M3 wczytany z: {le_3_path}")
        if le_precip_trained_for_m3:
            print(f"   Mapowanie klas M3 (wczytane): {dict(zip(le_precip_trained_for_m3.classes_, range(len(le_precip_trained_for_m3.classes_))))}")
    except Exception as e:
        print(f"   BŁĄD podczas wczytywania Modelu 3 lub LabelEncodera: {e}. Zostaną utworzone od nowa.")
        model_3 = None
        le_precip_trained_for_m3 = None

if model_3 is None or le_precip_trained_for_m3 is None or FORCE_RETRAIN:
    if FORCE_RETRAIN and model_3 is not None: print("   Wymuszono ponowny trening Modelu 3.")
    train_df_m3_subset = train_df[train_df['weather_category'].isin(precip_categories_user)].copy()
    if not train_df_m3_subset.empty:
        current_le_m3 = LabelEncoder() # Nowa instancja LabelEncodera
        y_train_m3 = current_le_m3.fit_transform(train_df_m3_subset['weather_category'])
        num_classes_m3 = len(current_le_m3.classes_)
        print(f"   Mapowanie klas M3 (trening): {dict(zip(current_le_m3.classes_, range(num_classes_m3)))}")
        
        model_3, returned_encoder = train_xgboost_model(train_df_m3_subset, y_train_m3, feature_lists_final['M3'],
                                                       "Model 3: Typy Opadów", 'multi:softmax', num_class=num_classes_m3,
                                                       use_smote=USE_SMOTE, label_encoder=current_le_m3)
        le_precip_trained_for_m3 = returned_encoder # Przypisz zwrócony (dopasowany) enkoder

        if model_3 and le_precip_trained_for_m3:
            try:
                model_3.save_model(model_3_path)
                joblib.dump(le_precip_trained_for_m3, le_3_path)
                print(f"   Model 3 zapisany do: {model_3_path}")
                print(f"   LabelEncoder dla M3 zapisany do: {le_3_path}")
            except Exception as e:
                print(f"   BŁĄD podczas zapisywania Modelu 3 lub LabelEncodera: {e}")
        elif model_3 is None: # Jeśli trening się nie udał
             le_precip_trained_for_m3 = None
    else:
        print("   BŁĄD: Brak danych treningowych dla Modelu 3 po filtrowaniu. Model nie zostanie wytrenowany.")
        model_3 = None
        le_precip_trained_for_m3 = None


# --- Model 4: Clear/Fair vs Cloudy/Overcast ---
model_4_path = os.path.join(MODEL_SAVE_DIR, "model_M4.json")
model_4 = None
if LOAD_MODELS_IF_EXIST and not FORCE_RETRAIN and os.path.exists(model_4_path):
    print(f"\n--- Wczytywanie Modelu 4 z pliku: {model_4_path} ---")
    try:
        model_4 = xgb.XGBClassifier()
        model_4.load_model(model_4_path)
        print("   Model 4 wczytany pomyślnie.")
    except Exception as e:
        print(f"   BŁĄD podczas wczytywania Modelu 4: {e}. Model zostanie wytrenowany od nowa.")
        model_4 = None
if model_4 is None or FORCE_RETRAIN:
    if FORCE_RETRAIN and model_4 is not None: print("   Wymuszono ponowny trening Modelu 4.")
    train_df_m4_subset = train_df[train_df['weather_category'].isin(['Clear/Fair', 'Cloudy/Overcast'])].copy()
    y_train_m4 = (train_df_m4_subset['weather_category'] == 'Cloudy/Overcast').astype(int) # 1 dla Cloudy, 0 dla Clear
    model_4, _ = train_xgboost_model(train_df_m4_subset, y_train_m4, feature_lists_final['M4'],
                                     "Model 4: Clear/Fair vs Cloudy/Overcast", 'binary:logistic', use_smote=USE_SMOTE)
    if model_4:
        try:
            model_4.save_model(model_4_path)
            print(f"   Model 4 zapisany do: {model_4_path}")
        except Exception as e:
            print(f"   BŁĄD podczas zapisywania Modelu 4: {e}")


# --- Predykcja Hierarchiczna na Zbiorze Testowym ---
print("\n--- Predykcja Hierarchiczna na Zbiorze Testowym ---")
models = {'M1': model_1, 'M2': model_2, 'M3': model_3, 'M4': model_4}
# Sprawdzenie, czy wszystkie modele są dostępne (wczytane lub wytrenowane)
models_available = True
for model_name, model_instance in models.items():
    if model_instance is None:
        print(f"BŁĄD KRYTYCZNY: Model {model_name} nie został wczytany ani wytrenowany. Przerywanie predykcji.")
        models_available = False
if not models_available:
    exit()

missing_cols_test = [col for col in all_unique_features_needed if col not in test_df.columns]
if missing_cols_test: print(f"BŁĄD: Brakuje kolumn w teście: {missing_cols_test}"); exit()
X_test_full = test_df[all_unique_features_needed].copy() # Użyj .copy() dla bezpieczeństwa
final_predictions_str = pd.Series(index=X_test_full.index, dtype=object, name='predicted_category')

print("Etap 1: Predykcja Opady/Brak (M1)...")
if feature_lists_final['M1']: # Sprawdź, czy są cechy dla M1
    pred_m1_binary = model_1.predict(X_test_full[feature_lists_final['M1']])
    indices_pred_p = X_test_full.index[pred_m1_binary == 1]
    indices_pred_np = X_test_full.index[pred_m1_binary == 0]
    print(f"   Przewidziano Opady: {len(indices_pred_p)}, Brak: {len(indices_pred_np)}.")
else:
    print("   BŁĄD: Brak cech dla Modelu 1. Pomijam Etap 1.")
    indices_pred_p = pd.Index([])
    indices_pred_np = X_test_full.index # Załóżmy brak opadów dla wszystkich, jeśli M1 nie działa

print("\nEtap 2: Predykcja Brak Opadów (M2 - Mgła vs Reszta)...")
indices_pred_fog = pd.Index([]); indices_pred_other_np = pd.Index([])
if not indices_pred_np.empty and feature_lists_final['M2']: # Sprawdź cechy M2
    X_test_np_subset = X_test_full.loc[indices_pred_np, feature_lists_final['M2']]
    pred_m2_binary = model_2.predict(X_test_np_subset)
    mask_fog = (pred_m2_binary == 1); mask_other = (pred_m2_binary == 0)
    indices_pred_fog = X_test_np_subset.index[mask_fog]
    indices_pred_other_np = X_test_np_subset.index[mask_other]
    final_predictions_str.loc[indices_pred_fog] = 'Fog'
    print(f"   Przewidziano Mgła: {len(indices_pred_fog)}.")
    print(f"   Pozostałe próbki bez opadów (nie-mgła): {len(indices_pred_other_np)}.")
elif not indices_pred_np.empty:
    print("   BŁĄD: Brak cech dla Modelu 2. Próbki 'Brak Opadów' nie będą dalej klasyfikowane przez M2.")
    indices_pred_other_np = indices_pred_np # Wszystkie próbki NP idą do M4
else: print("   Brak próbek 'Brak Opadów' z Etapu 1.")


print("\nEtap 3: Predykcja Opady (M3 - Typy Opadów)...")
if not indices_pred_p.empty and le_precip_trained_for_m3 is not None and feature_lists_final['M3']: # Sprawdź enkoder i cechy M3
    X_test_p_subset = X_test_full.loc[indices_pred_p, feature_lists_final['M3']]
    pred_m3_numeric = model_3.predict(X_test_p_subset)
    pred_m3_labels = le_precip_trained_for_m3.inverse_transform(pred_m3_numeric)
    final_predictions_str.loc[indices_pred_p] = pred_m3_labels
    print(f"   Przypisano typy dla {len(indices_pred_p)} próbek opadowych.")
elif not indices_pred_p.empty and le_precip_trained_for_m3 is None:
    print("   BŁĄD: Brak LabelEncodera (le_precip_trained_for_m3) dla Modelu 3. Próbki 'Opady' nie zostaną sklasyfikowane.")
elif not indices_pred_p.empty and not feature_lists_final['M3']:
    print("   BŁĄD: Brak cech dla Modelu 3. Próbki 'Opady' nie zostaną sklasyfikowane.")
else: print("   Brak próbek 'Opady' z Etapu 1.")

print("\nEtap 4: Predykcja Inne Bez Opadów (M4 - Clear/Fair vs Cloudy/Overcast)...")
if not indices_pred_other_np.empty and feature_lists_final['M4']: # Sprawdź cechy M4
    X_test_other_np_subset = X_test_full.loc[indices_pred_other_np, feature_lists_final['M4']]
    pred_m4_binary = model_4.predict(X_test_other_np_subset)
    # W Modelu 4: 0 to 'Clear/Fair', 1 to 'Cloudy/Overcast'
    mask_clear = (pred_m4_binary == 0); mask_cloudy = (pred_m4_binary == 1)
    indices_to_set_clear = X_test_other_np_subset.index[mask_clear]
    indices_to_set_cloudy = X_test_other_np_subset.index[mask_cloudy]
    final_predictions_str.loc[indices_to_set_clear] = 'Clear/Fair'
    final_predictions_str.loc[indices_to_set_cloudy] = 'Cloudy/Overcast'
    print(f"   Przypisano 'Clear/Fair' ({len(indices_to_set_clear)} próbek) lub 'Cloudy/Overcast' ({len(indices_to_set_cloudy)} próbek).")
elif not indices_pred_other_np.empty:
    print("   BŁĄD: Brak cech dla Modelu 4. Próbki 'Inne Bez Opadów' nie zostaną sklasyfikowane.")
else: print("   Brak próbek 'Inne Bez Opadów' z Etapu 2.")


missing_preds = final_predictions_str.isnull().sum()
if missing_preds > 0: print(f"\nOSTRZEŻENIE: {missing_preds} próbek nie otrzymało finalnej predykcji!")
final_predictions_str.dropna(inplace=True) # Usuń próbki, które mogły nie dostać predykcji (np. z powodu błędów w modelach)

# --- Ewaluacja Końcowa i Indywidualna ---
if final_predictions_str.empty:
    print("\nBŁĄD KRYTYCZNY: Brak jakichkolwiek finalnych predykcji. Nie można przeprowadzić ewaluacji.")
    exit()

common_index = y_test_actual_str.index.intersection(final_predictions_str.index)
if common_index.empty:
    print("\nBŁĄD KRYTYCZNY: Brak wspólnych indeksów między rzeczywistymi a przewidywanymi etykietami. Nie można przeprowadzić ewaluacji.")
    exit()

y_test_eval_str = y_test_actual_str.loc[common_index]
y_pred_eval_str = final_predictions_str.loc[common_index]

if len(y_test_eval_str) == 0: print("\nBŁĄD KRYTYCZNY: Brak wspólnych próbek do ewaluacji po dopasowaniu indeksów!"); exit()

print(f"\n--- === Ewaluacja Końcowa Modelu Hierarchicznego v6 (Wiedeń, {len(y_test_eval_str)} próbek) === ---")
overall_accuracy = accuracy_score(y_test_eval_str, y_pred_eval_str)
print(f"   Dokładność (Accuracy) Ogólna: {overall_accuracy:.4f}")
print("\n   Raport Klasyfikacji Ogólny:")
present_labels_overall = sorted(list(set(y_test_eval_str.unique()) | set(y_pred_eval_str.unique())))
# Upewnij się, że all_categories_user zawiera wszystkie present_labels_overall dla spójności macierzy
labels_for_cm_overall = sorted(list(set(all_categories_user) | set(present_labels_overall)))

print(classification_report(y_test_eval_str, y_pred_eval_str, labels=present_labels_overall, zero_division=0, digits=3))
print("\n   Macierz Pomyłek Ogólna:")
cm_overall = confusion_matrix(y_test_eval_str, y_pred_eval_str, labels=labels_for_cm_overall)
cm_overall_df = pd.DataFrame(cm_overall, index=labels_for_cm_overall, columns=labels_for_cm_overall)
# Filtruj macierz, aby pokazać tylko te kategorie, które faktycznie wystąpiły w teście lub predykcjach
cm_overall_df_filtered = cm_overall_df.loc[present_labels_overall, present_labels_overall]
# Usuń wiersze/kolumny z samymi zerami, jeśli takie powstały przez labels=labels_for_cm_overall
cm_overall_df_filtered = cm_overall_df_filtered.loc[(cm_overall_df_filtered.sum(axis=1) != 0), (cm_overall_df_filtered.sum(axis=0) != 0)]
print(cm_overall_df_filtered)

if VIZ_AVAILABLE and not cm_overall_df_filtered.empty:
    plt.figure(figsize=(max(8, len(cm_overall_df_filtered.columns)*1.2), max(6, len(cm_overall_df_filtered.index)*1)))
    sns.heatmap(cm_overall_df_filtered, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Macierz Pomyłek Ogólna - Model v6 (Wiedeń)\nRok testowy: {test_year} (Acc: {overall_accuracy:.3f})')
    plt.xlabel('Przewidywana Kategoria'); plt.ylabel('Rzeczywista Kategoria')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f"macierz_pomylek_OGOLNA_Wieden_rok{test_year}.png"), dpi=150, bbox_inches='tight') ### ZMIENIONE: Zapis do katalogu ###
    print(f"   Zapisano ogólną macierz pomyłek do pliku w katalogu: {MODEL_SAVE_DIR}")
    plt.show()

# --- Ewaluacja Indywidualnych Modeli ---
print("\n--- === Ewaluacja Modeli Składowych v6 (Wiedeń) === ---")

# Ewaluacja Modelu 1
if model_1 and feature_lists_final['M1']: # Sprawdź czy model istnieje i ma cechy
    print("\nEwaluacja Modelu 1: Opady vs Brak Opadów")
    y_test_m1_actual_str = y_test_actual_str.apply(lambda x: 'Opady' if x in precip_categories_user else 'Brak Opadów')
    X_test_m1_eval = test_df[feature_lists_final['M1']].copy() # Użyj .copy()
    common_m1 = y_test_m1_actual_str.index.intersection(X_test_m1_eval.index)
    if not common_m1.empty: evaluate_model(model_1, X_test_m1_eval.loc[common_m1], y_test_m1_actual_str.loc[common_m1], "Model 1: Opady vs Brak Opadów")
    else: print("   Brak wspólnych indeksów dla ewaluacji M1.")
else: print("   Model 1 lub jego cechy nie są dostępne. Pomijam ewaluację.")

# Ewaluacja Modelu 2
if model_2 and feature_lists_final['M2']:
    print("\nEwaluacja Modelu 2: Mgła vs Inne Bez Opadów")
    test_df_m2_subset_actual = test_df[test_df['weather_category'].isin(no_precip_categories_user)]
    if not test_df_m2_subset_actual.empty:
        y_test_m2_actual_str = test_df_m2_subset_actual['weather_category'].apply(lambda x: 'Mgła' if x == 'Fog' else 'Inne Bez Opadów')
        X_test_m2_eval = test_df_m2_subset_actual[feature_lists_final['M2']].copy()
        common_m2 = y_test_m2_actual_str.index.intersection(X_test_m2_eval.index)
        if not common_m2.empty: evaluate_model(model_2, X_test_m2_eval.loc[common_m2], y_test_m2_actual_str.loc[common_m2], "Model 2: Mgła vs Inne Bez Opadów")
        else: print("   Brak wspólnych indeksów dla ewaluacji M2.")
    else: print("   Brak danych testowych (po filtrowaniu) dla ewaluacji Modelu 2.")
else: print("   Model 2 lub jego cechy nie są dostępne. Pomijam ewaluację.")

# Ewaluacja Modelu 3
if model_3 and le_precip_trained_for_m3 and feature_lists_final['M3']: # Sprawdź też enkoder
    print("\nEwaluacja Modelu 3: Typy Opadów")
    test_df_m3_subset_actual = test_df[test_df['weather_category'].isin(precip_categories_user)]
    if not test_df_m3_subset_actual.empty:
        y_test_m3_actual_str = test_df_m3_subset_actual['weather_category']
        X_test_m3_eval = test_df_m3_subset_actual[feature_lists_final['M3']].copy()
        common_m3 = y_test_m3_actual_str.index.intersection(X_test_m3_eval.index)
        if not common_m3.empty: evaluate_model(model_3, X_test_m3_eval.loc[common_m3], y_test_m3_actual_str.loc[common_m3], "Model 3: Typy Opadów", label_encoder=le_precip_trained_for_m3)
        else: print("   Brak wspólnych indeksów dla ewaluacji M3.")
    else: print("   Brak danych testowych (po filtrowaniu) dla ewaluacji Modelu 3.")
else: print("   Model 3, jego LabelEncoder lub cechy nie są dostępne. Pomijam ewaluację.")

# Ewaluacja Modelu 4
if model_4 and feature_lists_final['M4']:
    print("\nEwaluacja Modelu 4: Clear/Fair vs Cloudy/Overcast")
    test_df_m4_subset_actual = test_df[test_df['weather_category'].isin(['Clear/Fair', 'Cloudy/Overcast'])]
    if not test_df_m4_subset_actual.empty:
        y_test_m4_actual_str = test_df_m4_subset_actual['weather_category'] # Bezpośrednio 'Clear/Fair' lub 'Cloudy/Overcast'
        X_test_m4_eval = test_df_m4_subset_actual[feature_lists_final['M4']].copy()
        common_m4 = y_test_m4_actual_str.index.intersection(X_test_m4_eval.index)
        if not common_m4.empty: evaluate_model(model_4, X_test_m4_eval.loc[common_m4], y_test_m4_actual_str.loc[common_m4], "Model 4: Clear/Fair vs Cloudy/Overcast")
        else: print("   Brak wspólnych indeksów dla ewaluacji M4.")
    else: print("   Brak danych testowych (po filtrowaniu) dla ewaluacji Modelu 4.")
else: print("   Model 4 lub jego cechy nie są dostępne. Pomijam ewaluację.")

print("\n--- Zakończono Skrypt Hierarchiczny XGBoost v6 (Wiedeń) ---")