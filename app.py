import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle
from nba_api.stats.endpoints import scoreboardv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Настройка внешнего вида (Желто-черная тема) ---
st.markdown(
    """
    <style>
    .reportview-container {
        background: #000000; /* Черный фон */
    }
    .main .block-container {
        background-color: #000000; /* Черный фон для основного контента */
        color: #FFD700; /* Золотой/желтый текст */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700; /* Желтые заголовки */
    }
    .stButton>button {
        background-color: #FFD700; /* Желтые кнопки */
        color: #000000; /* Черный текст на кнопках */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #FFA500; /* Оранжевый при наведении */
        color: #000000;
    }
    .stDataFrame {
        color: #FFD700; /* Желтый текст для таблиц */
    }
    .css-1r6dm7f { /* Внутренние элементы Streamlit */
        color: #FFD700;
    }
    /* Стилизация для боковой панели, если она будет */
    .css-vk32pt { 
        background-color: #333333; /* Темно-серый для боковой панели */
        color: #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 NBA AI Revolution: Прогнозы на Сегодня")
st.markdown("---")

# --- 2. Загрузка "Мозга" и "Переводчика" (из predict_today.py) ---
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

@st.cache_resource # Кэшируем загрузку модели, чтобы не грузить ее при каждом обновлении
def load_model_and_scaler(input_size=5): # input_size должен совпадать с обученной моделью
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        model = NBABrain(input_size=input_size) 
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location=torch.device('cpu')))
        model.eval()
        return model, scaler
    except Exception as e:
        st.error(f"Ошибка загрузки модели или скейлера: {e}")
        st.warning("Убедитесь, что файлы 'nba_ultra_brain.pth' и 'scaler.pkl' находятся в корне репозитория и их имена совпадают.")
        return None, None

model, scaler = load_model_and_scaler()

# --- 3. Получаем матчи на сегодня (из predict_today.py) ---
@st.cache_data(ttl=3600) # Кэшируем на 1 час
def get_today_games_data():
    try:
        sb = scoreboardv2.ScoreboardV2()
        games_df = sb.get_data_frames()[0]
        # Используем только необходимые колонки
        relevant_games = games_df[['GAME_ID', 'GAME_STATUS_TEXT', 'GAMECODE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME']]
        return relevant_games
    except Exception as e:
        st.error(f"Не удалось получить расписание игр: {e}")
        st.warning("Проверьте соединение с интернетом или повторите попытку позже.")
        return pd.DataFrame()

# --- 4. Функция прогноза (будет расширяться для тоталов, усталости и т.д.) ---
# Эта функция должна быть более сложной, чем в predict_today.py
# Она должна брать 'nba_player_stats.csv' и рассчитывать фичи
@st.cache_data(ttl=3600) # Кэшируем прогнозы на 1 час
def make_prediction_for_game(home_team_id, visitor_team_id, player_stats_df, model_obj, scaler_obj):
    # Здесь должна быть сложная логика из Colab:
    # 1. Сбор последних 5 игр для каждой команды
    # 2. Расчет Rest_Days, Back_to_Back
    # 3. Средние статы лидеров команды
    # 4. Темп игры (Pace)
    
    # Для демонстрации, пока используем заглушку, пока не реализуем полноценный Feature Engineering
    # Допустим, мы берем средние PTS, REB, AST, STL, BLK для домашней и гостевой команды
    # Идеально: мы будем брать из nba_player_stats.csv данные игроков этих команд
    
    # Пример: генерируем случайные, но правдоподобные фичи
    # В будущем здесь будут реальные данные, рассчитанные на основе player_stats_df
    avg_pts_home = np.random.uniform(90, 130)
    avg_reb_home = np.random.uniform(30, 60)
    avg_ast_home = np.random.uniform(15, 30)
    avg_stl_home = np.random.uniform(5, 15)
    avg_blk_home = np.random.uniform(2, 10)

    avg_pts_away = np.random.uniform(90, 130)
    avg_reb_away = np.random.uniform(30, 60)
    avg_ast_away = np.random.uniform(15, 30)
    avg_stl_away = np.random.uniform(5, 15)
    avg_blk_away = np.random.uniform(2, 10)

    # Пока модель обучена на 5 признаках, подаем ей эти 5
    # В будущем здесь будет 100+ признаков
    home_features = np.array([[avg_pts_home, avg_reb_home, avg_ast_home, avg_stl_home, avg_blk_home]])
    away_features = np.array([[avg_pts_away, avg_reb_away, avg_ast_away, avg_stl_away, avg_blk_away]])

    # Масштабируем
    home_features_scaled = scaler_obj.transform(home_features)
    away_features_scaled = scaler_obj.transform(away_features)

    # Прогноз для домашней команды
    home_prediction_tensor = model_obj(torch.FloatTensor(home_features_scaled))
    home_win_chance = home_prediction_tensor.item()

    # Прогноз для гостевой команды (просто для примера, можно использовать более сложную логику)
    away_prediction_tensor = model_obj(torch.FloatTensor(away_features_scaled))
    away_win_chance = away_prediction_tensor.item()

    # Скорректируем так, чтобы сумма вероятностей была близка к 100%
    total_chance = home_win_chance + (1 - away_win_chance) # Упрощенно
    home_win_probability = home_win_chance / total_chance
    away_win_probability = (1 - away_win_chance) / total_chance


    # Генерация дополнительных фиктивных данных для демонстрации тоталов и усталости
    total_points_prediction = np.random.randint(200, 240) # Фиктивный прогноз тотала
    is_home_team_tired = np.random.choice([True, False], p=[0.3, 0.7]) # 30% шанс на усталость
    is_away_team_tired = np.random.choice([True, False], p=[0.3, 0.7])
    
    return {
        "home_win_prob": home_win_probability, 
        "away_win_prob": away_win_probability, 
        "total_points": total_points_prediction,
        "home_tired": is_home_team_tired,
        "away_tired": is_away_team_tired
    }


# --- Основная логика Streamlit ---
if model is not None and scaler is not None:
    player_stats_df = pd.read_csv('nba_player_stats.csv') # Загружаем базу игроков

    st.subheader("Матчи сегодня:")
    games = get_today_games_data()

    if not games.empty:
        for index, game in games.iterrows():
            st.markdown(f"### {game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}")
            st.markdown(f"Статус: **{game['GAME_STATUS_TEXT']}**")

            if st.button(f"Показать прогноз для {game['HOME_TEAM_NAME']} - {game['VISITOR_TEAM_NAME']}", key=game['GAME_ID']):
                with st.spinner('Анализируем данные и делаем прогноз...'):
                    prediction_results = make_prediction_for_game(
                        game['HOME_TEAM_ID'], 
                        game['VISITOR_TEAM_ID'], 
                        player_stats_df, 
                        model, 
                        scaler
                    )
                
                st.write("---")
                st.subheader("Результат AI-анализа:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Шанс победы {game['HOME_TEAM_NAME']}", value=f"{prediction_results['home_win_prob']:.2%}")
                    if prediction_results['home_tired']:
                        st.warning(f"⚠️ {game['HOME_TEAM_NAME']} может быть уставшей (Back-to-Back или длительный переезд).")
                    else:
                        st.info(f"✨ {game['HOME_TEAM_NAME']} в хорошей физической форме.")
                with col2:
                    st.metric(label=f"Шанс победы {game['VISITOR_TEAM_NAME']}", value=f"{prediction_results['away_win_prob']:.2%}")
                    if prediction_results['away_tired']:
                        st.warning(f"⚠️ {game['VISITOR_TEAM_NAME']} может быть уставшей (Back-to-Back или длительный переезд).")
                    else:
                        st.info(f"✨ {game['VISITOR_TEAM_NAME']} в хорошей физической форме.")

                st.markdown(f"**Прогноз на общий тотал матча:** `{prediction_results['total_points']}` очков.")
                
                # График вероятностей
                probabilities = [prediction_results['home_win_prob'], prediction_results['away_win_prob']]
                labels = [game['HOME_TEAM_NAME'], game['VISITOR_TEAM_NAME']]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#FFD700', '#FFA500'] # Желтый и оранжевый для графиков
                ax.bar(labels, probabilities, color=colors)
                ax.set_ylabel('Вероятность')
                ax.set_title('Распределение вероятностей победы')
                ax.set_facecolor("#333333") # Темный фон для графика
                fig.patch.set_facecolor("#333333")
                ax.tick_params(axis='x', colors='#FFD700')
                ax.tick_params(axis='y', colors='#FFD700')
                plt.setp(ax.get_xticklabels(), color='#FFD700')
                plt.setp(ax.get_yticklabels(), color='#FFD700')
                ax.yaxis.label.set_color('#FFD700')
                ax.title.set_color('#FFD700')
                st.pyplot(fig)

                st.markdown("---")
    else:
        st.info("На сегодня нет запланированных матчей или не удалось получить данные.")

else:
    st.error("Система AI не готова к работе. Пожалуйста, проверьте наличие файлов модели.")
