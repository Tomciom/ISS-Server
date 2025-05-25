import sqlite3
import logging
import random
import datetime

DB_PATH = 'measurements.db'
HISTORICAL_DATA_MAC = "AA:BB:CC:11:22:33" # Ten sam MAC co poprzednio

# --- SEKCJA DANYCH PRZEPISANYCH ZE ZRZUTÓW EKRANU ---
# Dane dla 23, 24 i 25 Maja 2025
RAW_HISTORICAL_DATA_FROM_SCREENSHOTS = [
    # Dane dla 23 Maja 2025
    { "datetime_str": "2025-05-23 00:00", "temp_c": 11, "conditions": "Light rain. Passing clouds.", "wind_kmh": "8 km/h", "humidity_pct": "94%", "pressure_mbar": "1010 mbar" },
    { "datetime_str": "2025-05-23 00:30", "temp_c": 11, "conditions": "Light rain. Passing clouds.", "wind_kmh": "8 km/h", "humidity_pct": "94%", "pressure_mbar": "1010 mbar" },
    { "datetime_str": "2025-05-23 01:00", "temp_c": 10, "conditions": "Light rain. Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "94%", "pressure_mbar": "1010 mbar" },
    { "datetime_str": "2025-05-23 01:30", "temp_c": 10, "conditions": "Light rain. Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "94%", "pressure_mbar": "1011 mbar" },
    { "datetime_str": "2025-05-23 02:00", "temp_c": 10, "conditions": "Light rain. Passing clouds.", "wind_kmh": "6 km/h", "humidity_pct": "94%", "pressure_mbar": "1011 mbar" },
    { "datetime_str": "2025-05-23 02:30", "temp_c": 9, "conditions": "Light rain. Passing clouds.", "wind_kmh": "6 km/h", "humidity_pct": "94%", "pressure_mbar": "1011 mbar" },
    { "datetime_str": "2025-05-23 03:00", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "7 km/h", "humidity_pct": "87%", "pressure_mbar": "1011 mbar" }, # Zmiana conditions
    { "datetime_str": "2025-05-23 03:30", "temp_c": 9, "conditions": "Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "94%", "pressure_mbar": "1011 mbar" },
    { "datetime_str": "2025-05-23 04:00", "temp_c": 9, "conditions": "Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "94%", "pressure_mbar": "1012 mbar" },
    { "datetime_str": "2025-05-23 04:30", "temp_c": 9, "conditions": "Passing clouds.", "wind_kmh": "9 km/h", "humidity_pct": "94%", "pressure_mbar": "1012 mbar" },
    { "datetime_str": "2025-05-23 05:00", "temp_c": 9, "conditions": "Passing clouds.", "wind_kmh": "9 km/h", "humidity_pct": "94%", "pressure_mbar": "1012 mbar" },
    { "datetime_str": "2025-05-23 05:30", "temp_c": 8, "conditions": "Passing clouds. Broken clouds.", "wind_kmh": "9 km/h", "humidity_pct": "94%", "pressure_mbar": "1012 mbar" },
    { "datetime_str": "2025-05-23 06:00", "temp_c": 8, "conditions": "Partly sunny.", "wind_kmh": "7 km/h", "humidity_pct": "94%", "pressure_mbar": "1012 mbar" },
    { "datetime_str": "2025-05-23 06:30", "temp_c": 9, "conditions": "Partly sunny.", "wind_kmh": "9 km/h", "humidity_pct": "94%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-23 07:00", "temp_c": 9, "conditions": "Partly sunny.", "wind_kmh": "11 km/h", "humidity_pct": "94%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-23 07:30", "temp_c": 10, "conditions": "Partly sunny.", "wind_kmh": "11 km/h", "humidity_pct": "82%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-23 08:00", "temp_c": 9, "conditions": "Partly sunny.", "wind_kmh": "8 km/h", "humidity_pct": "82%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-23 08:30", "temp_c": 10, "conditions": "Partly sunny.", "wind_kmh": "7 km/h", "humidity_pct": "82%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-23 09:00", "temp_c": 10, "conditions": "Scattered clouds.", "wind_kmh": "11 km/h", "humidity_pct": "78%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-23 09:30", "temp_c": 10, "conditions": "Scattered clouds.", "wind_kmh": "8 km/h", "humidity_pct": "78%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 10:00", "temp_c": 10, "conditions": "Scattered clouds.", "wind_kmh": "11 km/h", "humidity_pct": "71%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 10:30", "temp_c": 10, "conditions": "Scattered clouds.", "wind_kmh": "11 km/h", "humidity_pct": "71%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 11:00", "temp_c": 11, "conditions": "Scattered clouds.", "wind_kmh": "11 km/h", "humidity_pct": "64%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 11:30", "temp_c": 11, "conditions": "Scattered clouds.", "wind_kmh": "8 km/h", "humidity_pct": "58%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 12:00", "temp_c": 11, "conditions": "Scattered clouds.", "wind_kmh": "6 km/h", "humidity_pct": "58%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 12:30", "temp_c": 12, "conditions": "Partly sunny.", "wind_kmh": "5 km/h", "humidity_pct": "54%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 13:00", "temp_c": 12, "conditions": "Partly sunny.", "wind_kmh": "4 km/h", "humidity_pct": "58%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 13:30", "temp_c": 12, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "55%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 14:00", "temp_c": 12, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "58%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 14:30", "temp_c": 12, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "58%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 15:00", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "51%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 15:30", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "51%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 16:00", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "5 km/h", "humidity_pct": "51%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 16:30", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "6 km/h", "humidity_pct": "51%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 17:00", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "51%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 17:30", "temp_c": 13, "conditions": "Partly sunny.", "wind_kmh": "8 km/h", "humidity_pct": "55%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 18:00", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "55%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 18:30", "temp_c": 13, "conditions": "Partly sunny.", "wind_kmh": "7 km/h", "humidity_pct": "63%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 19:00", "temp_c": 12, "conditions": "Partly sunny.", "wind_kmh": "8 km/h", "humidity_pct": "63%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 19:30", "temp_c": 12, "conditions": "Scattered clouds.", "wind_kmh": "9 km/h", "humidity_pct": "67%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-23 20:00", "temp_c": 12, "conditions": "Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "72%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-23 20:30", "temp_c": 11, "conditions": "Passing clouds.", "wind_kmh": "6 km/h", "humidity_pct": "72%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-23 21:00", "temp_c": 11, "conditions": "Passing clouds.", "wind_kmh": "8 km/h", "humidity_pct": "72%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-23 21:30", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "76%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-23 22:00", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "76%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-23 22:30", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "82%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-23 23:00", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "82%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-23 23:30", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "82%", "pressure_mbar": "1016 mbar" },

    # Dane dla 24 Maja 2025
    { "datetime_str": "2025-05-24 00:00", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "82%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 00:30", "temp_c": 10, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "87%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 01:00", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "87%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 01:30", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "87%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 02:00", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "87%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 02:30", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "87%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 03:00", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "5 km/h", "humidity_pct": "82%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 03:30", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "5 km/h", "humidity_pct": "82%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 04:00", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "6 km/h", "humidity_pct": "78%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 04:30", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "75%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 05:00", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "71%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 05:30", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "71%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 06:00", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "75%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 06:30", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "7 km/h", "humidity_pct": "82%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 07:00", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "8 km/h", "humidity_pct": "82%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 07:30", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "7 km/h", "humidity_pct": "82%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 08:00", "temp_c": 9, "conditions": "Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "78%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 08:30", "temp_c": 10, "conditions": "Passing clouds.", "wind_kmh": "5 km/h", "humidity_pct": "75%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 09:00", "temp_c": 10, "conditions": "Sunny.", "wind_kmh": "9 km/h", "humidity_pct": "70%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 09:30", "temp_c": 12, "conditions": "Passing clouds.", "wind_kmh": "7 km/h", "humidity_pct": "63%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 10:00", "temp_c": 12, "conditions": "Passing clouds.", "wind_kmh": "5 km/h", "humidity_pct": "59%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 10:30", "temp_c": 13, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "55%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 11:00", "temp_c": 13, "conditions": "Partly sunny.", "wind_kmh": "11 km/h", "humidity_pct": "55%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 11:30", "temp_c": 13, "conditions": "Partly sunny.", "wind_kmh": "9 km/h", "humidity_pct": "51%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 12:00", "temp_c": 14, "conditions": "Partly sunny.", "wind_kmh": "9 km/h", "humidity_pct": "51%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 12:30", "temp_c": 14, "conditions": "Partly sunny.", "wind_kmh": "13 km/h", "humidity_pct": "51%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 13:00", "temp_c": 14, "conditions": "Partly sunny.", "wind_kmh": "6 km/h", "humidity_pct": "51%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 13:30", "temp_c": 14, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "48%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 14:00", "temp_c": 15, "conditions": "Scattered clouds.", "wind_kmh": "No wind", "humidity_pct": "45%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 14:30", "temp_c": 15, "conditions": "Partly sunny.", "wind_kmh": "7 km/h", "humidity_pct": "45%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 15:00", "temp_c": 15, "conditions": "Partly sunny.", "wind_kmh": "9 km/h", "humidity_pct": "39%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 15:30", "temp_c": 15, "conditions": "Partly sunny.", "wind_kmh": "9 km/h", "humidity_pct": "42%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 16:00", "temp_c": 15, "conditions": "Partly sunny.", "wind_kmh": "6 km/h", "humidity_pct": "42%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-24 16:30", "temp_c": 15, "conditions": "Partly sunny.", "wind_kmh": "6 km/h", "humidity_pct": "39%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 17:00", "temp_c": 15, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "42%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 17:30", "temp_c": 15, "conditions": "Sunny.", "wind_kmh": "6 km/h", "humidity_pct": "39%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 18:00", "temp_c": 15, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "44%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 18:30", "temp_c": 14, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "51%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 19:00", "temp_c": 14, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "55%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 19:30", "temp_c": 13, "conditions": "Sunny.", "wind_kmh": "2 km/h", "humidity_pct": "58%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-24 20:00", "temp_c": 12, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "63%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 20:30", "temp_c": 11, "conditions": "Clear.", "wind_kmh": "No wind", "humidity_pct": "67%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 21:00", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "71%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 21:30", "temp_c": 9, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "71%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 22:00", "temp_c": 8, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "76%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 22:30", "temp_c": 8, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "81%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 23:00", "temp_c": 7, "conditions": "Clear.", "wind_kmh": "1 km/h", "humidity_pct": "81%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-24 23:30", "temp_c": 7, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "81%", "pressure_mbar": "1018 mbar" },

    # Dane dla 25 Maja 2025
    { "datetime_str": "2025-05-25 00:00", "temp_c": 7, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "81%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 00:30", "temp_c": 6, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "81%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 01:00", "temp_c": 6, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "87%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 01:30", "temp_c": 6, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "87%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 02:00", "temp_c": 5, "conditions": "Clear.", "wind_kmh": "4 km/h", "humidity_pct": "87%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 02:30", "temp_c": 5, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 03:00", "temp_c": 5, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 03:30", "temp_c": 5, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 04:00", "temp_c": 4, "conditions": "Clear.", "wind_kmh": "2 km/h", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 04:30", "temp_c": 4, "conditions": "Fog.", "wind_kmh": "2 km/h", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 05:00", "temp_c": 4, "conditions": "Fog.", "wind_kmh": "4 km/h", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 05:30", "temp_c": 5, "conditions": "Fog.", "wind_kmh": "No wind", "humidity_pct": "93%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 06:00", "temp_c": 5, "conditions": "Sunny.", "wind_kmh": "2 km/h", "humidity_pct": "87%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 06:30", "temp_c": 7, "conditions": "Sunny.", "wind_kmh": "2 km/h", "humidity_pct": "87%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 07:00", "temp_c": 8, "conditions": "Sunny.", "wind_kmh": "2 km/h", "humidity_pct": "82%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 07:30", "temp_c": 9, "conditions": "Sunny.", "wind_kmh": "4 km/h", "humidity_pct": "78%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 08:00", "temp_c": 11, "conditions": "Sunny.", "wind_kmh": "2 km/h", "humidity_pct": "77%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 08:30", "temp_c": 13, "conditions": "Sunny.", "wind_kmh": "2 km/h", "humidity_pct": "67%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 09:00", "temp_c": 14, "conditions": "Passing clouds.", "wind_kmh": "2 km/h", "humidity_pct": "63%", "pressure_mbar": "1017 mbar" },
    { "datetime_str": "2025-05-25 09:30", "temp_c": 14, "conditions": "Passing clouds.", "wind_kmh": "6 km/h", "humidity_pct": "51%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-25 10:00", "temp_c": 15, "conditions": "Passing clouds.", "wind_kmh": "5 km/h", "humidity_pct": "45%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-25 10:30", "temp_c": 15, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "45%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-25 11:00", "temp_c": 15, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "42%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-25 11:30", "temp_c": 16, "conditions": "Partly sunny.", "wind_kmh": "4 km/h", "humidity_pct": "39%", "pressure_mbar": "1016 mbar" },
    { "datetime_str": "2025-05-25 12:00", "temp_c": 16, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "36%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-25 12:30", "temp_c": 16, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "39%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-25 13:00", "temp_c": 17, "conditions": "Scattered clouds.", "wind_kmh": "9 km/h", "humidity_pct": "37%", "pressure_mbar": "1015 mbar" },
    { "datetime_str": "2025-05-25 13:30", "temp_c": 17, "conditions": "Passing clouds.", "wind_kmh": "4 km/h", "humidity_pct": "32%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-25 14:00", "temp_c": 18, "conditions": "Scattered clouds.", "wind_kmh": "9 km/h", "humidity_pct": "34%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-25 14:30", "temp_c": 17, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "42%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-25 15:00", "temp_c": 17, "conditions": "Scattered clouds.", "wind_kmh": "7 km/h", "humidity_pct": "34%", "pressure_mbar": "1014 mbar" },
    { "datetime_str": "2025-05-25 15:30", "temp_c": 18, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "32%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-25 16:00", "temp_c": 19, "conditions": "Scattered clouds.", "wind_kmh": "4 km/h", "humidity_pct": "32%", "pressure_mbar": "1013 mbar" },
    { "datetime_str": "2025-05-25 16:30", "temp_c": 19, "conditions": "Scattered clouds.", "wind_kmh": "9 km/h", "humidity_pct": "32%", "pressure_mbar": "1013 mbar" },
    # Koniec danych dla 25 maja o 16:30
]
# --- KONIEC SEKCJI DANYCH ZE ZRZUTÓW ---

def parse_transcribed_data(raw_data_list):
    """Konwertuje surowe przepisane dane na bardziej użyteczny format."""
    parsed_points = []
    for raw_point in raw_data_list:
        try:
            dt_obj = datetime.datetime.strptime(raw_point["datetime_str"], '%Y-%m-%d %H:%M')
            
            temp = int(raw_point["temp_c"]) # Temp jest int
            
            wind_str = raw_point["wind_kmh"].lower()
            if "no wind" in wind_str:
                wind = 0
            else:
                wind = int(wind_str.replace(" km/h", "").strip())
            
            humidity = int(raw_point["humidity_pct"].replace("%", "").strip())
            pressure = int(raw_point["pressure_mbar"].replace(" mbar", "").strip())

            parsed_points.append({
                "datetime": dt_obj,
                "temp_c": temp,
                "conditions_text": raw_point["conditions"],
                "wind_kmh": wind,
                "humidity_pct": humidity,
                "pressure_mbar": pressure
            })
        except Exception as e:
            logging.error(f"Błąd parsowania danych ze zrzutu dla wpisu {raw_point}: {e}")
            continue
            
    parsed_points.sort(key=lambda p: p["datetime"])
    return parsed_points

def map_conditions_to_db_schema(conditions_text, hour_of_day):
    """Mapuje tekstowy opis pogody na wartości sunshine i precipitation."""
    sunshine = 50.0 
    precipitation = 0.0

    conditions_lower = conditions_text.lower()
    is_night = hour_of_day < 6 or hour_of_day >= 21 # Przybliżone godziny nocne

    if "light rain" in conditions_lower or "drizzle" in conditions_lower:
        precipitation = round(random.uniform(0.1, 0.3), 2)
    elif "rain" in conditions_lower:
        precipitation = round(random.uniform(0.4, 1.5), 2)
    elif "showers" in conditions_lower:
         precipitation = round(random.uniform(0.5, 2.0), 2)

    if "clear" in conditions_lower:
        sunshine = 0.0 if is_night else round(random.uniform(85.0, 100.0), 1)
    elif "sunny" in conditions_lower:
        sunshine = round(random.uniform(80.0, 100.0), 1)
        if is_night:
            logging.warning(f"Warunki 'sunny' o godzinie {hour_of_day}:00 (noc). Ustawiam niskie nasłonecznienie.")
            sunshine = round(random.uniform(0.0, 10.0),1)
    elif "partly sunny" in conditions_lower or \
         "scattered clouds" in conditions_lower or \
         "passing clouds" in conditions_lower or \
         "broken clouds" in conditions_lower: # Dodano "broken clouds"
        sunshine = 0.0 if is_night else round(random.uniform(30.0, 70.0), 1)
    elif "cloudy" in conditions_lower or "overcast" in conditions_lower:
        sunshine = 0.0 if is_night else round(random.uniform(10.0, 30.0), 1)
    elif "fog" in conditions_lower or "mist" in conditions_lower:
        sunshine = round(random.uniform(0.0, 15.0), 1)
        if random.random() < 0.05 and precipitation == 0.0:
            precipitation = round(random.uniform(0.0, 0.1), 2)

    if precipitation > 0.0:
        sunshine = min(sunshine, round(random.uniform(0.0, 25.0), 1))

    return sunshine, precipitation

def save_mac_to_db(username, mac_address):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE username = ?", (username,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"User '{username}' not found.")
            user_id = result[0]
            cur.execute(
                "SELECT 1 FROM user_boards WHERE user_id = ? AND mac_address = ?",
                (user_id, mac_address)
            )
            if cur.fetchone():
                logging.info(f"Device {mac_address} already associated with user {username}") # Zmieniono na info
                return False
            cur.execute(
                "INSERT INTO user_boards (user_id, mac_address) VALUES (?, ?)",
                (user_id, mac_address)
            )
            conn.commit()
            logging.info(f"Device {mac_address} successfully associated with user {username}")
            return True
    except sqlite3.IntegrityError:
        logging.error(f"IntegrityError when adding device {mac_address} for user {username}. Duplicate entry?")
        return False
    except ValueError as ve:
        logging.error(f"Error in save_mac_to_db: {ve}")
        raise ve
    except Exception as e:
        logging.error(f"An unexpected error occurred in save_mac_to_db: {e}")
        raise e

def save_measurement(data):
    if not data or 'mac_address' not in data or 'server_timestamp' not in data:
        logging.error("Invalid data provided for save_measurement. 'mac_address' and 'server_timestamp' are required.")
        raise ValueError("Data must contain 'mac_address' and 'server_timestamp'.")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                '''
                INSERT INTO measurements (
                    mac_address, server_timestamp, temperature, pressure,
                    humidity, sunshine, wind_speed, precipitation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    data.get('mac_address'), data.get('server_timestamp'),
                    data.get('temperature'), data.get('pressure'),
                    data.get('humidity'), data.get('sunshine'),
                    data.get('wind_speed'), data.get('precipitation'),
                )
            )
            conn.commit()
    except sqlite3.Error as e:
        sql_statement = "N/A"
        if 'cur' in locals() and hasattr(cur, 'statement'):
            sql_statement = cur.statement if cur.statement else "N/A"
        logging.error(f"SQLite error saving measurement: {e} - SQL: {sql_statement} - Data: {data}")
        raise e
    except Exception as e:
        logging.error(f"General error saving measurement: {e} - Data: {data}")
        raise e


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Przygotowanie bazy danych
    try:
        with sqlite3.connect(DB_PATH) as conn_setup:
            conn_setup.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL)')
            conn_setup.execute('CREATE TABLE IF NOT EXISTS user_boards (user_id INTEGER, mac_address TEXT, PRIMARY KEY (user_id, mac_address), FOREIGN KEY (user_id) REFERENCES users(id))')
            conn_setup.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, mac_address TEXT, server_timestamp TIMESTAMP,
                    temperature REAL, pressure REAL, humidity REAL,
                    sunshine REAL, wind_speed REAL, precipitation REAL
                )''')
            try:
                conn_setup.execute("INSERT INTO users (id, username) VALUES (?, ?)", (1, 'admin'))
                logging.info("Użytkownik 'admin' z ID=1 dodany (lub już istniał z tym ID).")
            except sqlite3.IntegrityError:
                conn_setup.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", ('admin',))
                logging.info("Użytkownik 'admin' gotowy (ID zostanie przydzielone automatycznie lub już istnieje).")
            conn_setup.commit()
            logging.info("Tabele 'users', 'user_boards', 'measurements' gotowe.")
    except Exception as e_setup:
        logging.error(f"Nie udało się przygotować bazy danych: {e_setup}")
        exit()

    # 2. Przypisanie MAC do użytkownika "admin"
    admin_user = "admin"
    logging.info(f"Próba przypisania MAC {HISTORICAL_DATA_MAC} do użytkownika '{admin_user}'...")
    try:
        save_mac_to_db(admin_user, HISTORICAL_DATA_MAC)
    except Exception as e_assign:
        logging.error(f"Problem podczas przypisywania MAC: {e_assign}")


    # 3. Przetwarzanie i zapisywanie danych ze zrzutów ekranu
    logging.info("Przetwarzanie danych historycznych ze zrzutów ekranu...")
    
    parsed_historical_points = parse_transcribed_data(RAW_HISTORICAL_DATA_FROM_SCREENSHOTS)

    if not parsed_historical_points:
        logging.warning("Brak danych ze zrzutów do przetworzenia. Sprawdź listę RAW_HISTORICAL_DATA_FROM_SCREENSHOTS.")
    else:
        logging.info(f"Znaleziono {len(parsed_historical_points)} punktów danych ze zrzutów do przetworzenia (interwały 30-minutowe).")
        num_records_generated_from_screenshots = 0
        log_interval_db = 5000 # Loguj postęp co X zapisanych rekordów (5-sekundowych)

        for i in range(len(parsed_historical_points)):
            current_source_point = parsed_historical_points[i]
            current_source_dt = current_source_point["datetime"]

            if i + 1 < len(parsed_historical_points):
                next_source_dt = parsed_historical_points[i+1]["datetime"]
            else:
                # Ostatni punkt danych. Jego wartości obowiązują przez standardowy interwał 30 minut.
                next_source_dt = current_source_dt + datetime.timedelta(minutes=30) 
            
            logging.debug(f"Przetwarzanie danych dla okresu od {current_source_dt} do {next_source_dt} na podstawie wpisu: {current_source_point['conditions_text']}")

            loop_time_for_db = current_source_dt
            while loop_time_for_db < next_source_dt:
                temp_c = current_source_point["temp_c"]
                pressure_hpa = current_source_point["pressure_mbar"]
                humidity_0_1 = round(current_source_point["humidity_pct"] / 100.0, 2)
                wind_speed_kmh = current_source_point["wind_kmh"]
                
                sunshine_pct, precip_mm = map_conditions_to_db_schema(
                    current_source_point["conditions_text"],
                    loop_time_for_db.hour
                )

                db_data_packet = {
                    'mac_address': HISTORICAL_DATA_MAC,
                    'server_timestamp': loop_time_for_db.strftime('%Y-%m-%d %H:%M:%S'),
                    'temperature': float(temp_c),
                    'pressure': float(pressure_hpa),
                    'humidity': float(humidity_0_1),
                    'sunshine': float(sunshine_pct),
                    'wind_speed': float(wind_speed_kmh),
                    'precipitation': float(precip_mm),
                }

                try:
                    save_measurement(data=db_data_packet)
                    num_records_generated_from_screenshots += 1
                    if num_records_generated_from_screenshots % log_interval_db == 0:
                        logging.info(f"Zapisano {num_records_generated_from_screenshots} rekordów do DB (dane ze zrzutów)...")
                except Exception as e_hist_save:
                    logging.error(f"Błąd przy zapisie rekordu DB dla {loop_time_for_db} (dane ze zrzutów): {e_hist_save}")

                loop_time_for_db += datetime.timedelta(seconds=5)

        logging.info(f"Zakończono przetwarzanie danych ze zrzutów. Całkowita liczba zapisanych rekordów: {num_records_generated_from_screenshots}")

    # 4. Weryfikacja
    try:
        with sqlite3.connect(DB_PATH) as conn_verify:
            cur_verify = conn_verify.cursor()
            logging.info(f"Ostatnie 5 zapisów w 'measurements' dla MAC {HISTORICAL_DATA_MAC}:")
            cur_verify.execute(
                "SELECT id, mac_address, server_timestamp, temperature, sunshine, precipitation FROM measurements WHERE mac_address = ? ORDER BY server_timestamp DESC LIMIT 5",
                (HISTORICAL_DATA_MAC,)
            )
            rows = cur_verify.fetchall()
            if rows:
                for row in rows:
                    logging.info(f"Odczytano: ID={row[0]}, MAC={row[1]}, Timestamp={row[2]}, Temp={row[3]}, Sun={row[4]}, Precip={row[5]}")
            else:
                logging.warning(f"Nie znaleziono zapisów dla MAC {HISTORICAL_DATA_MAC}.")
    except Exception as e_verify:
        logging.error(f"Błąd podczas weryfikacji danych z bazy: {e_verify}")