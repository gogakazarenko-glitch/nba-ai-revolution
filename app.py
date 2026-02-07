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

# --- 2. СТИЛЬ (ЧЕРНЫЙ И ЗОЛОТОЙ) ---
st.set_page_config(page_title="NBA AI", page_icon="🏀", layout="wide")
st.markdown("<style>.stApp { background-color: #000000; color: #FFD700; }</style>", unsafe_allow_html=True)

# --- 3. МОДЕЛЬ ---
class NBARegressionModel(nn.Module):
    def __init__(self, input_size):
        super(NBARegressionModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.win = nn.Linear(128, 1)
        self.total = nn.Linear(128, 1)
        self.spread = nn.Linear(128, 1)

    def forward(self, x):
        features = self.main(x)
        return torch.sigmoid(self.win(features)), self.total(features), self.spread(features)

# --- 4. ЗАГРУЗКА ---
@st.cache_resource
def load_assets():
    try:
        model = NBARegressionModel(113)
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location='cpu', weights_only=False))
        model.eval()
        with open('scaler_final.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except Exception as e:
        return None, str(e)

model_data = load_assets()
model, preprocessor = model_data if isinstance(model_data[0], NBARegressionModel) else (None, None)

# --- 5. ПОЛУЧЕНИЕ МАТЧЕЙ (УЛУЧШЕННОЕ) ---
def get_games():
    try:
        from nba_api.stats.live.endpoints import scoreboard
        games = scoreboard.ScoreBoard().get_dict()['scoreboard']['games']
        return games
    except:
        try:
            sb = scoreboardv2.ScoreboardV2()
            return sb.get_dict()['resultSets'][0]['rowSet']
        except:
            return []

# --- 6. ИНТЕРФЕЙС ---
st.title("🏀 NBA AI REVOLUTION")

if model:
    st.success("ИИ готов. Анализируем текущий ростер...")
    raw_games = get_games()
    
    if raw_games:
        for g in raw_games:
            # Пытаемся достать названия команд из разных форматов API
            try:
                home = g.get('homeTeam', {}).get('teamName') or g[6]
                away = g.get('awayTeam', {}).get('teamName') or g[7]
                gid = g.get('gameId') or g[2]
            except:
                continue

            with st.expander(f"➡️ {away} @ {home}"):
                if st.button(f"Сделать прогноз", key=gid):
                    # Заглушка данных для теста нейронки
                    dummy = np.zeros((1, 113))
                    X = torch.tensor(preprocessor.transform(dummy), dtype=torch.float32)
                    with torch.no_grad():
                        p, t, s = model(X)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Победа дома", f"{p.item():.1%}")
                    col2.metric("Тотал", f"{t.item():.1f}")
                    col3.metric("Фора", f"{s.item():.1f}")
    else:
        st.info("Матчи загружаются... Если пусто, попробуйте обновить страницу через минуту.")
else:
    st.error(f"Ошибка системы: {model_data[1]}")
