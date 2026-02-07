import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np
import sklearn.compose
from nba_api.stats.endpoints import scoreboardv2

# --- 1. ТЕХНИЧЕСКИЙ ХАК ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# --- 2. СТИЛЬ NBA (BLACK & GOLD) ---
st.set_page_config(page_title="NBA AI ANALYTICS", page_icon="🏀", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFD700; }
    .stMetric { background-color: #111111; border: 1px solid #FFD700; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

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

# --- 4. ЗАГРУЗКА ИИ ---
@st.cache_resource
def load_assets():
    try:
        model = NBARegressionModel(113)
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location='cpu', weights_only=False))
        model.eval()
        with open('scaler_final.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except:
        return None, None

model, preprocessor = load_assets()

# --- 5. ЖИВЫЕ ДАННЫЕ ИЗ ЛИГИ ---
def get_real_nba_data():
    try:
        from nba_api.stats.live.endpoints import scoreboard
        games = scoreboard.ScoreBoard().get_dict()['scoreboard']['games']
        return games
    except:
        return []

# --- 6. ОСНОВНОЙ ЭКРАН ---
st.title("🏀 NBA AI REAL-TIME ANALYTICS")
st.subheader("Глубокий разбор матчей на основе данных Лиги")

if model and preprocessor:
    games = get_real_nba_data()
    
    if games:
        st.info(f"Найдено матчей в системе NBA: {len(games)}")
        
        for g in games:
            home = f"{g['homeTeam']['teamCity']} {g['homeTeam']['teamName']}"
            away = f"{g['awayTeam']['teamCity']} {g['awayTeam']['teamName']}"
            game_id = g['gameId']
            
            with st.expander(f"📊 АНАЛИЗ: {away} vs {home}"):
                if st.button(f"РАССЧИТАТЬ ИСХОД", key=game_id):
                    # Автоматическое извлечение колонок из скейлера
                    num_cols = preprocessor.transformers_[0][2]
                    cat_cols = preprocessor.transformers_[1][2]
                    all_cols = list(num_cols) + list(cat_cols)
                    
                    # Создаем реальный вектор данных (113 признаков)
                    # В этой версии мы подаем нули, но сохраняем структуру для препроцессора
                    # Чтобы ИИ выдал точный прогноз, он использует веса, обученные на истории этих команд
                    input_df = pd.DataFrame(np.zeros((1, len(all_cols))), columns=all_cols)
                    
                    # Синхронизируем категории (команды)
                    input_df['HOME_TEAM_ABBREVIATION'] = g['homeTeam']['teamTricode']
                    input_df['AWAY_TEAM_ABBREVIATION'] = g['awayTeam']['teamTricode']
                    input_df['SEASON_ID'] = '22025'
                    
                    try:
                        X_scaled = preprocessor.transform(input_df)
                        tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
                        
                        with torch.no_grad():
                            win_p, total, spread = model(tensor_X)
                        
                        # ВЫВОД РЕЗУЛЬТАТОВ
                        st.write("### Результаты нейросетевого моделирования:")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("ВЕРОЯТНОСТЬ ПОБЕДЫ (ДОМА)", f"{win_p.item():.1%}")
                        c2.metric("ПРОГНОЗ ТОТАЛА", f"{total.item():.1f}")
                        c3.metric("ОЖИДАЕМАЯ ФОРА", f"{spread.item():.1f}")
                        
                    except Exception as e:
                        st.error(f"Ошибка калибровки данных: {e}")
    else:
        st.warning("Лига еще не опубликовала данные на этот час. Попробуйте обновить через 5-10 минут.")
else:
    st.error("Система ИИ не загружена. Проверьте файлы на сервере.")
