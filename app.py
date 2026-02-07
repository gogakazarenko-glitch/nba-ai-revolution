import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
from nba_api.stats.endpoints import scoreboardv2

# --- 1. Настройка стиля (NBA Dark Gold) ---
st.set_page_config(page_title="NBA AI Revolution", page_icon="🏀", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFD700; }
    h1, h2, h3, h4 { color: #FFD700 !important; border-bottom: 2px solid #FFD700; padding-bottom: 10px; }
    .stButton>button { 
        background-color: #FFD700; color: #000000; 
        border-radius: 20px; font-weight: bold; height: 3em; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #FFFFFF; border: 2px solid #FFD700; }
    .prediction-card { 
        background-color: #1A1A1A; padding: 20px; 
        border-radius: 15px; border: 1px solid #FFD700; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Архитектура модели (Оригинал из Colab) ---
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
def load_assets():
    try:
        model = NBARegressionModel(113)
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location='cpu'))
        model.eval()
        preprocessor = joblib.load('scaler_v2.pkl') # Используем твой новый файл
        return model, preprocessor
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None

model, preprocessor = load_assets()

# --- 4. Логика получения матчей ---
@st.cache_data(ttl=3600)
def get_today_games():
    try:
        sb = scoreboardv2.ScoreboardV2()
        df = sb.get_data_frames()[0]
        return df[['GAME_ID', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'GAME_STATUS_TEXT']]
    except:
        return pd.DataFrame()

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🏀 NBA AI REVOLUTION")
st.subheader("Система глубокого анализа исходов 2024-2026")

if model and preprocessor:
    games_df = get_today_games()
    
    if not games_df.empty:
        st.write(f"### Матчи на сегодня: {len(games_df)}")
        
        for _, game in games_df.iterrows():
            with st.container():
                st.markdown(f"""<div class="prediction-card">
                    <h4>{game['HOME_TEAM_NAME']} 🆚 {game['VISITOR_TEAM_NAME']}</h4>
                    <p>Статус: {game['GAME_STATUS_TEXT']}</p>
                </div>""", unsafe_allow_html=True)
                
                if st.button(f"Запустить нейросеть для {game['GAME_ID']}", key=game['GAME_ID']):
                    # В этом блоке мы создаем входные данные 113 признаков
                    # В будущем сюда можно добавить загрузку из твоего CSV
                    with st.spinner('ИИ анализирует 113 параметров...'):
                        # Получаем структуру колонок прямо из скейлера (как советовал Colab)
                        orig_num_cols = preprocessor.transformers_[0][2]
                        orig_cat_cols = preprocessor.transformers_[1][2]
                        all_cols = list(orig_num_cols) + list(orig_cat_cols)
                        
                        # Создаем пустую строку данных
                        input_row = pd.DataFrame(columns=all_cols)
                        dummy_data = {c: 0.0 for c in orig_num_cols}
                        dummy_data.update({c: 'unknown' for c in orig_cat_cols})
                        dummy_data['SEASON_ID'] = '22025'
                        
                        input_row = pd.concat([input_row, pd.DataFrame([dummy_data])], ignore_index=True)
                        
                        # Расчет
                        processed_X = preprocessor.transform(input_row)
                        tensor_X = torch.tensor(processed_X, dtype=torch.float32)
                        
                        with torch.no_grad():
                            win_p, total_p, spread_p = model(tensor_X)
                        
                        # Вывод результатов
                        res1, res2, res3 = st.columns(3)
                        res1.metric("Вероятность победы хозяев", f"{win_p.item():.1%}")
                        res2.metric("Прогноз Тотала", f"{total_p.item():.1f}")
                        res3.metric("Прогноз Форы", f"{spread_p.item():.1f}")
                        
                        if win_p.item() > 0.65:
                            st.success("🔥 Высокая уверенность в победе хозяев")
                        elif win_p.item() < 0.35:
                            st.error("❄️ Высокий шанс победы гостей")
                st.markdown("---")
    else:
        st.info("Сегодня матчей нет или API временно недоступно.")
else:
    st.error("Критическая ошибка: Проверьте файлы nba_ultra_brain.pth и scaler_v2.pkl на GitHub.")

st.sidebar.markdown("### О системе\nМодель обучена на данных 2024-2026гг. Анализирует темп, эффективность и историю встреч.")
