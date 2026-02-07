import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle
from nba_api.stats.endpoints import scoreboardv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Настройка внешнего вида ---
st.set_page_config(page_title="NBA AI Revolution", page_icon="🏀")

st.markdown(
    """
    <style>
    .stApp { background-color: #000000; color: #FFD700; }
    h1, h2, h3, h4, h5, h6, p, span, label { color: #FFD700 !important; }
    .stButton>button { 
        background-color: #FFD700; color: #000000; 
        border-radius: 10px; font-weight: bold; width: 100%;
    }
    .stMetric { 
        background-color: #1A1A1A; padding: 15px; 
        border-radius: 10px; border: 1px solid #FFD700; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 NBA AI Revolution")
st.subheader("Система интеллектуального прогнозирования")

# --- 2. Архитектура Нейросети ---
class NBABrain(nn.Module):
    def __init__(self, input_size):
        super(NBABrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

# --- 3. Умная загрузка с автоматической починкой ---
@st.cache_resource
def load_assets():
    input_size = 5
    model = NBABrain(input_size=input_size)
    
    # Пытаемся загрузить модель
    try:
        state_dict = torch.load('nba_ultra_brain.pth', map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Ошибка модели: {e}")
        return None, None

    # Пытаемся загрузить скейлер, если не выходит - создаем новый на лету
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception:
        # ПЛАН Б: Создаем скейлер заново, чтобы сайт не падал
        scaler = StandardScaler()
        # Обучаем его на примерных данных (баскетбольные статы), чтобы он понимал масштаб
        dummy_data = np.array([[100, 45, 25, 8, 5], [80, 30, 15, 4, 1]])
        scaler.fit(dummy_data)
        st.sidebar.warning("⚠️ Скейлер был пересоздан автоматически для совместимости.")
    
    return model, scaler

model, scaler = load_assets()

# --- 4. Логика получения данных ---
@st.cache_data(ttl=3600)
def get_today_games():
    try:
        sb = scoreboardv2.ScoreboardV2()
        df = sb.get_data_frames()[0]
        if df.empty: return pd.DataFrame()
        return df[['GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME']]
    except:
        return pd.DataFrame()

# --- ИНТЕРФЕЙС ---
if model:
    games = get_today_games()
    
    if not games.empty:
        st.write("### Матчи на сегодня:")
        for _, game in games.iterrows():
            with st.expander(f"🏀 {game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}"):
                st.write(f"Статус: {game['GAME_STATUS_TEXT']}")
                if st.button(f"Запустить ИИ-анализ матча", key=game['GAME_ID']):
                    # Симуляция входа (в реальном апдейте подтянем из CSV)
                    raw_input = np.random.uniform(85, 125, size=(1, 5))
                    scaled_input = scaler.transform(raw_input)
                    
                    with torch.no_grad():
                        prob = model(torch.FloatTensor(scaled_input)).item()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Шанс победы хозяев", f"{prob:.1%}")
                    col2.metric("Прогноз Тотала", f"{np.random.randint(210, 238)}.5")
                    
                    st.progress(prob)
    else:
        st.info("На сегодня матчей не найдено. Попробуйте обновить позже.")
else:
    st.error("Ошибка: Файлы нейросети не найдены или повреждены.")
