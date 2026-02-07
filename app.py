import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np
import sklearn.compose
from nba_api.stats.endpoints import scoreboardv2

# --- 1. ХАК ДЛЯ СОВМЕСТИМОСТИ (Исправляет ошибку _RemainderColsList) ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# --- 2. НАСТРОЙКА СТИЛЯ NBA ---
st.set_page_config(page_title="NBA AI Revolution", page_icon="🏀", layout="wide")
st.markdown("<style>.stApp { background-color: #000000; color: #FFD700; }</style>", unsafe_allow_html=True)

# --- 3. АРХИТЕКТУРА МОДЕЛИ ---
class NBARegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(NBARegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.output_home_win = nn.Linear(hidden_size, 1)
        self.output_total_points = nn.Linear(hidden_size, 1)
        self.output_point_spread = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        return torch.sigmoid(self.output_home_win(x)), self.output_total_points(x), self.output_point_spread(x)

# --- 4. ЗАГРУЗКА МОДЕЛИ И СКЕЙЛЕРА ---
@st.cache_resource
def load_all():
    try:
        model = NBARegressionModel(113)
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location='cpu', weights_only=False))
        model.eval()
        
        # Загрузка через pickle по инструкции
        with open('scaler_final.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None

model, preprocessor = load_all()

# --- 5. ИНТЕРФЕЙС ---
st.title("🏀 NBA AI REVOLUTION")
if model and preprocessor:
    st.success("Система ИИ онлайн! Анализ 2024-2026 активен.")
    
    # Кнопка для теста (потом добавим авто-загрузку матчей)
    if st.button("Рассчитать прогноз на сегодня"):
        st.info("ИИ обрабатывает данные... (Матчи появятся здесь)")
else:
    st.warning("Критическая ошибка файлов. Проверьте scaler_final.pkl и nba_ultra_brain.pth")
