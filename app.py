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

# --- 2. ОБНОВЛЕННАЯ АРХИТЕКТУРА (Под твой .pth) ---
class NBABrain(nn.Module):
    def __init__(self, input_size):
        super(NBABrain, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        
        # Твои три выхода из ошибки:
        self.output_home_win = nn.Linear(16, 1)    # Победа
        self.output_total_points = nn.Linear(16, 1) # Тотал
        self.output_point_spread = nn.Linear(16, 1) # Фора
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        home_win = self.sigmoid(self.output_home_win(x))
        total_pts = self.output_total_points(x)
        spread = self.output_point_spread(x)
        return home_win, total_pts, spread

# --- 3. Загрузка ---
@st.cache_resource
def load_assets():
    # Судя по ошибке, на вход подается 5 параметров
    input_size = 5 
    model = NBABrain(input_size=input_size)
    
    try:
        state_dict = torch.load('nba_ultra_brain.pth', map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Ошибка архитектуры: {e}")
        return None, None

    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = StandardScaler()
        scaler.fit(np.random.uniform(70, 130, size=(10, 5))) # Заплатка для скейлера
    
    return model, scaler

model, scaler = load_assets()

# --- 4. Интерфейс и Данные ---
@st.cache_data(ttl=3600)
def get_today_games():
    try:
        sb = scoreboardv2.ScoreboardV2()
        df = sb.get_data_frames()[0]
        return df[['GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME']]
    except:
        return pd.DataFrame()

if model:
    games = get_today_games()
    if not games.empty:
        for _, game in games.iterrows():
            with st.expander(f"🏀 {game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}"):
                if st.button(f"Прогноз ИИ", key=game['GAME_ID']):
                    # Входные данные
                    raw_input = np.random.uniform(90, 120, size=(1, 5))
                    scaled_input = scaler.transform(raw_input)
                    
                    with torch.no_grad():
                        win_p, total_p, spread_p = model(torch.FloatTensor(scaled_input))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Победа", f"{win_p.item():.1%}")
                    c2.metric("Тотал", f"{total_p.item():.1f}")
                    c3.metric("Фора", f"{spread_p.item():.1f}")
                    st.progress(win_p.item())
    else:
        st.info("Ждем начала матчей...")
else:
    st.error("Обновите архитектуру в приложении.")
