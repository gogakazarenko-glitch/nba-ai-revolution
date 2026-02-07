import pandas as pd
import torch
import torch.nn as nn
import pickle
from nba_api.stats.endpoints import scoreboardv2

# 1. Загрузка "Мозга" и "Переводчика"
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

# Загружаем скейлер и модель
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Предположим, модель была обучена на 5 признаках (PTS, REB, AST, STL, BLK)
model = NBABrain(input_size=5) 
model.load_state_dict(torch.load('nba_ultra_brain.pth'))
model.eval()

# 2. Получаем матчи на сегодня
def get_today_games():
    sb = scoreboardv2.ScoreboardV2()
    games = sb.get_data_frames()[0]
    return games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]

# 3. Функция прогноза
def predict():
    games = get_today_games()
    stats = pd.read_csv('nba_player_stats.csv')
    
    print("--- РЕВОЛЮЦИОННЫЙ ПРОГНОЗ NBA ---")
    for _, game in games.iterrows():
        # Тут ИИ берет средние статы команд из твоего CSV
        # (Упрощенная логика для примера)
        sample_input = torch.tensor([[20.0, 5.0, 5.0, 1.0, 1.0]]) # Тестовые данные
        prediction = model(sample_input).item()
        
        win_chance = round(prediction * 100, 2)
        print(f"Матч {game['GAME_ID']}: Шанс победы хозяев: {win_chance}%")

if __name__ == "__main__":
    predict()
