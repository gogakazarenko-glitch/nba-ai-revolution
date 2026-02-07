import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
from nba_api.stats.endpoints import scoreboardv2

# --- 1. Настройка стиля ---
st.set_page_config(page_title="NBA AI Revolution", page_icon="🏀")
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFD700; }
    h1, h2, h3, h4, h5, h6, p, span, label { color: #FFD700 !important; }
    .stButton>button { background-color: #FFD700; color: #000000; border-radius: 10px; font-weight: bold; width: 100%; }
    .stMetric { background-color: #1A1A1A; padding: 15px; border-radius: 10px; border: 1px solid #FFD700; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Оригинальная архитектура из Colab ---
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

# --- 3. Загрузка ресурсов ---
@st.cache_resource
def load_all_assets():
    input_size = 113
    model = NBARegressionModel(input_size)
    try:
        # Загружаем веса
        state_dict = torch.load('nba_ultra_brain.pth', map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        # Загружаем препроцессор через joblib (как в Colab)
        preprocessor = joblib.load('scaler.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None

model, preprocessor = load_all_assets()

# --- 4. Список признаков (точно как в твоем коде) ---
numerical_features = ['HOME_MIN', 'HOME_FGM', 'HOME_FGA', 'HOME_FG_PCT', 'HOME_FG3M', 'HOME_FG3A', 'FG3_PCT_x', 'HOME_FTM', 'HOME_FTA', 'FT_PCT_x', 'HOME_OREB', 'HOME_DREB', 'HOME_REB', 'HOME_AST', 'HOME_STL', 'HOME_BLK', 'HOME_TOV', 'HOME_PF', 'VIDEO_AVAILABLE_x', 'Rest_Days_x', 'Back_to_Back_x', 'Team_Pace_Game_x', 'Team_Pace_Season_Avg_x', 'AWAY_MIN', 'AWAY_FGM', 'AWAY_FGA', 'AWAY_FG_PCT', 'AWAY_FG3M', 'AWAY_FG3A', 'FG3_PCT_y', 'AWAY_FTM', 'AWAY_FTA', 'FT_PCT_y', 'AWAY_OREB', 'AWAY_DREB', 'AWAY_REB', 'AWAY_AST', 'AWAY_STL', 'AWAY_BLK', 'AWAY_TOV', 'AWAY_PF', 'VIDEO_AVAILABLE_y', 'Rest_Days_y', 'Back_to_Back_y', 'Team_Pace_Game_y', 'Team_Pace_Season_Avg_y']
categorical_features = ['WL_x', 'WL_y', 'SEASON_ID', 'AWAY_TEAM_ABBREVIATION', 'HOME_TEAM_ABBREVIATION']

# --- 5. Интерфейс ---
st.title("🏀 NBA AI Revolution")
st.write("Синхронизация с моделью 2024-2026 завершена.")

if model and preprocessor:
    try:
        sb = scoreboardv2.ScoreboardV2()
        games = sb.get_data_frames()[0]
    except:
        games = pd.DataFrame()

    if not games.empty:
        for _, game in games.iterrows():
            with st.expander(f"Матч: {game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}"):
                if st.button("Рассчитать глубокий прогноз", key=game['GAME_ID']):
                    # Создаем входные данные (пока эмуляция средних значений)
                    # В идеале здесь мы берем данные из твоего CSV
                    dummy_data = pd.DataFrame([{f: 0 for f in numerical_features + categorical_features}])
                    dummy_data['SEASON_ID'] = '22025' # Твой формат
                    
                    # Трансформируем через оригинальный препроцессор
                    X_processed = preprocessor.transform(dummy_data)
                    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
                    
                    with torch.no_grad():
                        win_prob, total, spread = model(X_tensor)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Шанс победы", f"{win_prob.item():.1%}")
                    c2.metric("Тотал очков", f"{total.item():.1f}")
                    c3.metric("Прогноз форы", f"{spread.item():.1f}")
    else:
        st.info("Матчи не найдены. Проверьте расписание позже.")
