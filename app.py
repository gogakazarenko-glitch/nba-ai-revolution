import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle
from nba_api.stats.endpoints import scoreboardv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# --- 1. Настройка внешнего вида (Желто-черная тема) ---
st.set_page_config(page_title="NBA AI Revolution", page_icon="🏀")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #FFD700;
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #FFD700 !important;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #000000;
        border-radius: 10px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        color: #000000;
    }
    .stMetric {
        background-color: #1A1A1A;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 NBA AI Revolution: Прогнозы на Сегодня")
st.markdown("---")

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

# --- 3. Умная загрузка модели и скейлера ---
@st.cache_resource
def load_assets(input_size=5):
    try:
        # Загрузка скейлера
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Загрузка модели с флагом безопасности для новых версий Torch
        model = NBABrain(input_size=input_size)
        state_dict = torch.load('nba_ultra_brain.pth', map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, scaler
    except Exception as e:
        st.error(f"Критическая ошибка загрузки: {e}")
        return None, None

model, scaler = load_assets()

# --- 4. Получение данных ---
@st.cache_data(ttl=3600)
def get_today_games():
    try:
        sb = scoreboardv2.ScoreboardV2()
        df = sb.get_data_frames()[0]
        return df[['GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    except:
        return pd.DataFrame()

# --- 5. Логика ИИ-прогноза ---
def make_ai_prediction(model_obj, scaler_obj):
    # Генерация входных данных на основе твоих параметров из Colab
    # В будущем здесь будет подтяжка из nba_player_stats.csv
    raw_data = np.random.uniform(90, 120, size=(1, 5)) 
    scaled_data = scaler_obj.transform(raw_data)
    
    with torch.no_grad():
        prob = model_obj(torch.FloatTensor(scaled_data)).item()
    
    return {
        "win_prob": prob,
        "total": np.random.randint(215, 235),
        "fatigue": np.random.choice(["Свежие", "Уставшие (B2B)"], p=[0.7, 0.3])
    }

# --- ИНТЕРФЕЙС ---
if model and scaler:
    games = get_today_games()
    
    if not games.empty:
        for _, game in games.iterrows():
            with st.container():
                st.subheader(f"🏟 {game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}")
                
                if st.button(f"Анализировать матч {game['GAME_ID'][-3:]}"):
                    res = make_ai_prediction(model, scaler)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Победа Хозяев", f"{res['win_prob']:.1%}")
                    with col2:
                        st.metric("Прогноз Тотала", res['total'])
                    with col3:
                        st.metric("Состояние", res['fatigue'])
                    
                    # Маленький график
                    fig, ax = plt.subplots(figsize=(4, 1))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    val = res['win_prob']
                    ax.barh(['Шанс'], [val], color='#FFD700')
                    ax.barh(['Шанс'], [1], color='#333333', left=[0], zorder=0)
                    ax.axis('off')
                    st.pyplot(fig)
                st.markdown("---")
    else:
        st.warning("Матчи на сегодня еще не загружены. Проверьте позже!")
else:
    st.info("Ожидание загрузки файлов ИИ...")
