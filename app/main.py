# app/main.py - Сервис для предсказания
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import os

# 1. Загружаем модель (учитывая структуру папок)
# Определяем путь к файлу относительно текущего скрипта
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sber_auto_model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model_bundle = pickle.load(file)
    model = model_bundle['model']

# 2. Инициализация FastAPI
app = FastAPI(title="SberAuto Prediction Service")


# 3. Описание входных данных
class SessionInput(BaseModel):
    session_id: str
    visit_date: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str = None
    utm_keyword: str = None
    device_category: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


# 4. Эндпоинт для предсказания
@app.post("/predict")
def predict(data: SessionInput):
    # 1. Преобразование входящих данных в DataFrame
    df = pd.DataFrame([data.dict()])

    # 2. Обработка даты и создание временных признаков
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df['visit_month'] = df['visit_date'].dt.month.fillna(0).astype(int)
    df['day_of_week'] = df['visit_date'].dt.dayofweek.fillna(0).astype(int)

    # --- Создание признака 'is_weekend' ---
    # Если день недели 5 (суббота) или 6 (воскресенье), то признак равен 1
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Обработка разрешения экрана
    res = str(data.device_screen_resolution)
    if 'x' in res:
        parts = res.split('x')
        df['screen_width'] = int(parts[0]) if parts[0].isdigit() else 0
        df['screen_height'] = int(parts[1]) if parts[1].isdigit() else 0
    else:
        df['screen_width'], df['screen_height'] = 0, 0

    # 4. Обработка источников из социальных сетей
    soc_list = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df['is_social'] = df['utm_source'].apply(lambda x: 1 if x in soc_list else 0)

    # 5. Приведение типов данных к категориальным (для корректной работы модели)
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype('category')

    # 6. Выполнение предсказания моделью
    y_pred = model.predict(df)

    return {
        "session_id": data.session_id,
        "prediction": int(y_pred[0])
    }

# 5. Тестовый эндпоинт для проверки работоспособности
@app.get("/status")
def status():
    return {"status": "Model is ready and running!"}