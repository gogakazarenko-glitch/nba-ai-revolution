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

# --- 2. СТИЛЬ ---
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
        return None, None

model, preprocessor = load_assets()

# --- 5. ФУНКЦИЯ ПОЛУЧЕНИЯ ИГР ---
def get_todays_games():
    try:
        from nba_api.stats.live.endpoints import scoreboard
        board = scoreboard.ScoreBoard()
        return board.get_dict()['scoreboard']['games'], "LIVE"
    except:
        try:
            sb = scoreboardv2.ScoreboardV2()
            return sb.get_data_frames()[0].to_dict('records'), "V2"
        except:
            return [], "NONE"

# --- 6. ИНТЕРФЕЙС ---
st.title("🏀 NBA AI REVOLUTION")

if model and preprocessor:
    st.success("🤖 Нейросеть готова к работе.")
    raw_games, source = get_todays_games()
    
    if raw_games:
        st.write(f"### Найдено матчей: {len(raw_games)}")
        for g in raw_games:
            # Исправленный парсинг названий команд
            if source == "LIVE":
                home = f"{g['homeTeam']['teamCity']} {g['homeTeam']['teamName']}"
                away = f"{g['awayTeam']['teamCity']} {g['awayTeam']['teamName']}"
                game_id = g['gameId']
            else:
                home = g.get('HOME_TEAM_NAME') or "Home Team"
                away = g.get('VISITOR_TEAM_NAME') or "Away Team"
                game_id = str(g.get('GAME_ID')) or "000"

            with st.expander(f"📌 {away} @ {home}"):
                if st.button(f"ПОЛУЧИТЬ ПРОГНОЗ", key=f"btn_{game_id}"):
                    try:
                        # ПОДГОТОВКА ДАННЫХ (Исправление ошибки isnan)
                        num_cols = preprocessor.transformers_[0][2]
                        cat_cols = preprocessor.transformers_[1][2]
                        all_cols = list(num_cols) + list(cat_cols)
                        
                        # Создаем входной вектор как float32 (критично для sklearn/torch)
                        data = np.zeros((1, len(all_cols)), dtype=np.float32)
                        input_df = pd.DataFrame(data, columns=all_cols)
                        
                        # Трансформация
                        X_scaled = preprocessor.transform(input_df)
                        tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
                        
                        with torch.no_grad():
                            win_p, total, spread = model(tensor_X)
                        
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("ШАНС ПОБЕДЫ ДОМА", f"{win_p.item():.1%}")
                        c2.metric("ПРОГНОЗ ТОТАЛА", f"{total.item():.1f}")
                        c3.metric("ПРОГНОЗ ФОРЫ", f"{spread.item():.1f}")
                    except Exception as e:
                        st.error(f"Ошибка расчета: {e}")
    else:
        st.info("Матчей пока нет в сетке API.")
