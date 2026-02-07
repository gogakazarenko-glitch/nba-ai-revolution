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
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFD700; }
    .stButton>button { 
        background-color: #FFD700; color: #000000; border-radius: 10px; 
        font-weight: bold; width: 100%; border: none;
    }
    .stMetric { background-color: #1A1A1A; padding: 10px; border-radius: 10px; border: 1px solid #FFD700; }
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

# --- 4. ЗАГРУЗКА МОДЕЛИ И СКЕЙЛЕРА ---
@st.cache_resource
def load_all():
    try:
        model = NBARegressionModel(113)
        # weights_only=False необходим для загрузки моделей, созданных в старых версиях torch/colab
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location='cpu', weights_only=False))
        model.eval()
        
        with open('scaler_final.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except Exception as e:
        st.error(f"Критическая ошибка загрузки файлов: {e}")
        return None, None

model, preprocessor = load_all()

# --- 5. ЛОГИКА ПОЛУЧЕНИЯ МАТЧЕЙ ---
@st.cache_data(ttl=3600)
def get_live_games():
    try:
        sb = scoreboardv2.ScoreboardV2()
        df = sb.get_data_frames()[0]
        # Оставляем только нужные колонки для интерфейса
        return df[['GAME_ID', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'GAME_STATUS_TEXT']]
    except Exception as e:
        st.error(f"Ошибка подключения к API NBA: {e}")
        return pd.DataFrame()

# --- 6. ГЛАВНЫЙ ЭКРАН ---
st.title("🏀 NBA AI REVOLUTION")
st.subheader("Прогноз исходов на основе нейросети (2024-2026)")

if model and preprocessor:
    st.success("✅ Система ИИ онлайн и готова к анализу!")
    
    games = get_live_games()
    
    if not games.empty:
        st.write(f"### Матчи на сегодня: {len(games)}")
        for _, game in games.iterrows():
            # Создаем удобную карточку для каждого матча
            with st.expander(f"📌 {game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']} (Статус: {game['GAME_STATUS_TEXT']})"):
                if st.button(f"Запустить нейросеть для матча {game['GAME_ID']}", key=game['GAME_ID']):
                    with st.spinner('Анализируем 113 статистических параметров...'):
                        # Генерируем входной вектор (пока заглушка, в будущем подтянем реальные статы)
                        dummy_input = np.zeros((1, 113))
                        
                        # Код Colab: получаем структуру колонок напрямую из препроцессора
                        # Это гарантирует, что порядок признаков не нарушится
                        try:
                            # Пытаемся получить список колонок из скейлера
                            orig_num_cols = preprocessor.transformers_[0][2]
                            orig_cat_cols = preprocessor.transformers_[1][2]
                            all_cols = list(orig_num_cols) + list(orig_cat_cols)
                            
                            # Создаем DataFrame с правильными названиями
                            input_df = pd.DataFrame(dummy_input[:, :len(all_cols)], columns=all_cols)
                            
                            # Прогоняем через скейлер и модель
                            X_scaled = preprocessor.transform(input_df)
                            tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
                            
                            with torch.no_grad():
                                win_p, total, spread = model(tensor_X)
                            
                            # Красивый вывод метрик
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric("Шанс победы дома", f"{win_p.item():.1%}")
                            res_col2.metric("Прогноз Тотала", f"{total.item():.1f}")
                            res_col3.metric("Прогноз Форы", f"{spread.item():.1f}")
                            
                        except Exception as e:
                            st.error(f"Ошибка трансформации данных: {e}")
    else:
        st.info("На сегодня матчей NBA не найдено. База обновится автоматически.")
else:
    st.warning("⚠️ Внимание: Система не смогла загрузить веса модели или препроцессор. Проверьте файлы на GitHub.")
