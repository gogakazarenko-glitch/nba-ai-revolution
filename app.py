import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib 
import pickle 
import numpy as np

# --- 1. Re-define the NBARegressionModel class ---
# This class must be identical to the one used for training
class NBARegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(NBARegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layers for each task
        self.output_home_win = nn.Linear(hidden_size, 1) # Binary classification
        self.output_total_points = nn.Linear(hidden_size, 1) # Regression
        self.output_point_spread = nn.Linear(hidden_size, 1) # Regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)

        home_win_output = torch.sigmoid(self.output_home_win(x))
        total_points_output = self.output_total_points(x)
        point_spread_output = self.output_point_spread(x)

        return home_win_output, total_points_output, point_spread_output

# --- 2. Define numerical and categorical features (must match training) ---
numerical_features = [
    'HOME_MIN', 'HOME_FGM', 'HOME_FGA', 'HOME_FG_PCT', 'HOME_FG3M', 'HOME_FG3A', 'FG3_PCT_x', 
    'HOME_FTM', 'HOME_FTA', 'FT_PCT_x', 'HOME_OREB', 'HOME_DREB', 'HOME_REB', 'HOME_AST', 
    'HOME_STL', 'HOME_BLK', 'HOME_TOV', 'HOME_PF', 'VIDEO_AVAILABLE_x', 'Rest_Days_x', 
    'Back_to_Back_x', 'Team_Pace_Game_x', 'Team_Pace_Season_Avg_x',
    'AWAY_MIN', 'AWAY_FGM', 'AWAY_FGA', 'AWAY_FG_PCT', 'AWAY_FG3M', 'AWAY_FG3A', 'FG3_PCT_y', 
    'AWAY_FTM', 'AWAY_FTA', 'FT_PCT_y', 'AWAY_OREB', 'AWAY_DREB', 'AWAY_REB', 'AWAY_AST', 
    'AWAY_STL', 'AWAY_BLK', 'AWAY_TOV', 'AWAY_PF', 'VIDEO_AVAILABLE_y', 'Rest_Days_y', 
    'Back_to_Back_y', 'Team_Pace_Game_y', 'Team_Pace_Season_Avg_y'
]
categorical_features = ['WL_x', 'WL_y', 'SEASON_ID', 'AWAY_TEAM_ABBREVIATION', 'HOME_TEAM_ABBREVIATION']

# --- 3. Load the trained model and preprocessor ---
@st.cache_resource # Cache resource to avoid reloading on every rerun
def load_model_and_preprocessor():
    input_size = 113 # This should match the actual input size used during training
    model = NBARegressionModel(input_size)
    try:
        model.load_state_dict(torch.load('nba_ultra_brain.pth', map_location=torch.device('cpu')))
        # Изменение здесь: загружаем scaler_final.pkl с помощью pickle
        with open('scaler_final.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        model.eval()
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model or preprocessor files not found. Make sure 'nba_ultra_brain.pth' and 'scaler_final.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        st.stop()

model, preprocessor = load_model_and_preprocessor()

# --- 4. Streamlit UI elements ---
st.title('NBA Game Outcome Predictor')
st.markdown('Enter details for a hypothetical NBA match to get predictions.')

# Input fields for HOME TEAM
st.header('Home Team Statistics')
home_team_name = st.selectbox('Home Team Abbreviation', options=['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'], key='home_abbr')
home_min = st.number_input('Home Team Avg MIN (per game)', min_value=240, max_value=240, value=240, key='home_min_input')
home_fgm = st.number_input('Home Team Avg FGM', min_value=0, value=40, key='home_fgm_input')
home_fga = st.number_input('Home Team Avg FGA', min_value=0, value=85, key='home_fga_input')
home_fg_pct = st.number_input('Home Team Avg FG%', min_value=0.0, max_value=1.0, value=0.470, format="%.3f", key='home_fg_pct_input')
home_fg3m = st.number_input('Home Team Avg FG3M', min_value=0, value=10, key='home_fg3m_input')
home_fg3a = st.number_input('Home Team Avg FG3A', min_value=0, value=30, key='home_fg3a_input')
home_fg3_pct = st.number_input('Home Team Avg FG3%', min_value=0.0, max_value=1.0, value=0.333, format="%.3f", key='home_fg3_pct_input')
home_ftm = st.number_input('Home Team Avg FTM', min_value=0, value=15, key='home_ftm_input')
home_fta = st.number_input('Home Team Avg FTA', min_value=0, value=20, key='home_fta_input')
home_ft_pct = st.number_input('Home Team Avg FT%', min_value=0.0, max_value=1.0, value=0.750, format="%.3f", key='home_ft_pct_input')
home_oreb = st.number_input('Home Team Avg OREB', min_value=0, value=10, key='home_oreb_input')
home_dreb = st.number_input('Home Team Avg DREB', min_value=0, value=30, key='home_dreb_input')
home_reb = st.number_input('Home Team Avg REB', min_value=0, value=40, key='home_reb_input')
home_ast = st.number_input('Home Team Avg AST', min_value=0, value=25, key='home_ast_input')
home_stl = st.number_input('Home Team Avg STL', min_value=0, value=7, key='home_stl_input')
home_blk = st.number_input('Home Team Avg BLK', min_value=0, value=5, key='home_blk_input')
home_tov = st.number_input('Home Team Avg TOV', min_value=0, value=12, key='home_tov_input')
home_pf = st.number_input('Home Team Avg PF', min_value=0, value=20, key='home_pf_input')
home_wl = st.selectbox('Home Team Win/Loss in last game', options=['W', 'L'], key='home_wl_input')
home_rest_days = st.number_input('Home Team Rest Days', min_value=0, value=2, key='home_rest_days_input')
home_b2b = st.radio('Home Team Back-to-Back?', options=[0, 1], index=0, key='home_b2b_input')
home_pace_game = st.number_input('Home Team Pace (Current Game)', min_value=0, value=100, key='home_pace_game_input')
home_pace_season_avg = st.number_input('Home Team Pace (Season Avg)', min_value=0, value=100, key='home_pace_season_avg_input')



# Input fields for AWAY TEAM
st.header('Away Team Statistics')
away_team_name = st.selectbox('Away Team Abbreviation', options=['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'], key='away_abbr')
away_min = st.number_input('Away Team Avg MIN (per game)', min_value=240, max_value=240, value=240, key='away_min_input')
away_fgm = st.number_input('Away Team Avg FGM', min_value=0, value=38, key='away_fgm_input')
away_fga = st.number_input('Away Team Avg FGA', min_value=0, value=88, key='away_fga_input')
away_fg_pct = st.number_input('Away Team Avg FG%', min_value=0.0, max_value=1.0, value=0.430, format="%.3f", key='away_fg_pct_input')
away_fg3m = st.number_input('Away Team Avg FG3M', min_value=0, value=12, key='away_fg3m_input')
away_fg3a = st.number_input('Away Team Avg FG3A', min_value=0, value=35, key='away_fg3a_input')
away_fg3_pct = st.number_input('Away Team Avg FG3%', min_value=0.0, max_value=1.0, value=0.340, format="%.3f", key='away_fg3_pct_input')
away_ftm = st.number_input('Away Team Avg FTM', min_value=0, value=18, key='away_ftm_input')
away_fta = st.number_input('Away Team Avg FTA', min_value=0, value=25, key='away_fta_input')
away_ft_pct = st.number_input('Away Team Avg FT%', min_value=0.0, max_value=1.0, value=0.720, format="%.3f", key='away_ft_pct_input')
away_oreb = st.number_input('Away Team Avg OREB', min_value=0, value=8, key='away_oreb_input')
away_dreb = st.number_input('Away Team Avg DREB', min_value=0, value=28, key='away_dreb_input')
away_reb = st.number_input('Away Team Avg REB', min_value=0, value=36, key='away_reb_input')
away_ast = st.number_input('Away Team Avg AST', min_value=0, value=22, key='away_ast_input')
away_stl = st.number_input('Away Team Avg STL', min_value=0, value=6, key='away_stl_input')
away_blk = st.number_input('Away Team Avg BLK', min_value=0, value=4, key='away_blk_input')
away_tov = st.number_input('Away Team Avg TOV', min_value=0, value=14, key='away_tov_input')
away_pf = st.number_input('Away Team Avg PF', min_value=0, value=22, key='away_pf_input')
away_wl = st.selectbox('Away Team Win/Loss in last game', options=['W', 'L'], key='away_wl_input')
away_rest_days = st.number_input('Away Team Rest Days', min_value=0, value=1, key='away_rest_days_input')
away_b2b = st.radio('Away Team Back-to-Back?', options=[0, 1], index=1, key='away_b2b_input')
away_pace_game = st.number_input('Away Team Pace (Current Game)', min_value=0, value=98, key='away_pace_game_input')
away_pace_season_avg = st.number_input('Away Team Pace (Season Avg)', min_value=0, value=99, key='away_pace_season_avg_input')


# General game info
st.header('Game Information')
season_id = st.selectbox('Season ID', options=['22023', '22024', '22025'], index=2)
video_available_x = st.radio('Video Available (Home Game)', options=[0, 1], index=1, key='video_x')
video_available_y = st.radio('Video Available (Away Game)', options=[0, 1], index=1, key='video_y')



# --- 5. Generate predictions on button click ---
if st.button('Predict Game Outcome'):
    # Create a dictionary for new game data, matching feature names
    new_game_data_dict = {
        'HOME_TEAM_ABBREVIATION': home_team_name,
        'HOME_MIN': home_min,
        'HOME_FGM': home_fgm,
        'HOME_FGA': home_fga,
        'HOME_FG_PCT': home_fg_pct,
        'HOME_FG3M': home_fg3m,
        'HOME_FG3A': home_fg3a,
        'FG3_PCT_x': home_fg3_pct, # Renamed during preprocessing
        'HOME_FTM': home_ftm,
        'HOME_FTA': home_fta,
        'FT_PCT_x': home_ft_pct, # Renamed during preprocessing
        'HOME_OREB': home_oreb,
        'HOME_DREB': home_dreb,
        'HOME_REB': home_reb,
        'HOME_AST': home_ast,
        'HOME_STL': home_stl,
        'HOME_BLK': home_blk,
        'HOME_TOV': home_tov,
        'HOME_PF': home_pf,
        'WL_x': home_wl, # Home team win/loss in last game
        'Rest_Days_x': float(home_rest_days),
        'Back_to_Back_x': float(home_b2b),
        'Team_Pace_Game_x': float(home_pace_game),
        'Team_Pace_Season_Avg_x': float(home_pace_season_avg),
        'VIDEO_AVAILABLE_x': float(video_available_x),

        'AWAY_TEAM_ABBREVIATION': away_team_name,
        'AWAY_MIN': away_min,
        'AWAY_FGM': away_fgm,
        'AWAY_FGA': away_fga,
        'AWAY_FG_PCT': away_fg_pct,
        'AWAY_FG3M': away_fg3m,
        'AWAY_FG3A': away_fg3a,
        'FG3_PCT_y': away_fg3_pct, # Renamed during preprocessing
        'AWAY_FTM': away_ftm,
        'AWAY_FTA': away_fta,
        'FT_PCT_y': away_ft_pct, # Renamed during preprocessing
        'AWAY_OREB': away_oreb,
        'AWAY_DREB': away_dreb,
        'AWAY_REB': away_reb,
        'AWAY_AST': away_ast,
        'AWAY_STL': away_stl,
        'AWAY_BLK': away_blk,
        'AWAY_TOV': away_tov,
        'AWAY_PF': away_pf,
        'WL_y': away_wl, # Away team win/loss in last game
        'Rest_Days_y': float(away_rest_days),
        'Back_to_Back_y': float(away_b2b),
        'Team_Pace_Game_y': float(away_pace_game),
        'Team_Pace_Season_Avg_y': float(away_pace_season_avg),
        'VIDEO_AVAILABLE_y': float(video_available_y),

        'SEASON_ID': season_id
    }

    # Ensure all numerical features are floats
    for k, v in new_game_data_dict.items():
        if k in numerical_features:
            new_game_data_dict[k] = float(v)

    # Create DataFrame in the correct column order
    # First, get all feature columns that the preprocessor expects
    # This requires knowing the exact order of columns `preprocessor.fit_transform` was called on.
    # A safer way is to ensure all expected columns are present and then reorder if necessary.
    
    # For simplicity, create a dummy DataFrame with all required columns and then populate
    # with user input. The order of columns in `numerical_features` + `categorical_features` 
    # should align with the order `preprocessor` expects.
    # The best practice is to reconstruct the original DataFrame's columns before dropping targets.
    
    # Let's assume the original training DataFrame columns (before dropping targets and IDs) were:
    # (numerical features then categorical features)
    # If the preprocessor was created with X = df_games.drop(columns=all_excluded_cols, errors='ignore')
    # then the order of X's columns is critical.
    # From previous notebook cells, the preprocessor was fitted on X.columns in the order they appeared.
    
    # Get all column names from the preprocessor's transformers
    original_numerical_cols = preprocessor.transformers_[0][2] # These are the numerical features
    original_categorical_cols = preprocessor.transformers_[1][2] # These are the categorical features
    
    all_expected_cols = list(original_numerical_cols) + list(original_categorical_cols)
    
    # Create a DataFrame for prediction, ensuring column order matches training data
    # Fill with placeholder (e.g., 0) then update with user input
    input_df = pd.DataFrame(columns=all_expected_cols)
    input_df = pd.concat([input_df, pd.DataFrame([new_game_data_dict])], ignore_index=True)

    # Ensure correct data types for processing
    for col in original_numerical_cols:
        if col in input_df.columns: # Check if column exists
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    for col in original_categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    
    # Critical: Handle potential missing columns in user input (though we tried to avoid it)
    # For Streamlit, it's safer to ensure all expected columns are present, even if some are default/0
    missing_cols = set(all_expected_cols) - set(input_df.columns)
    for c in missing_cols:
        # Default value for missing numerical features could be 0, for categorical 'unknown' or empty string
        if c in original_numerical_cols:
            input_df[c] = 0.0 # Or a more appropriate default/mean
        elif c in original_categorical_cols:
            input_df[c] = '' # OneHotEncoder handles 'ignore' for unknown categories
    
    # Reorder columns to match the training data expected by the preprocessor
    input_df = input_df[all_expected_cols]
    
    try:
        # Preprocess the input data
        X_new_processed = preprocessor.transform(input_df)
        X_new_tensor = torch.tensor(X_new_processed, dtype=torch.float32)

        # Generate predictions
        with torch.no_grad():
            home_win_pred_raw, total_points_pred_raw, point_spread_pred_raw = model(X_new_tensor)

        # Post-process and display results
        home_win_probability = home_win_pred_raw.item()
        home_win_predicted = "Home Team Wins" if home_win_probability > 0.5 else "Away Team Wins"
        total_points_predicted = total_points_pred_raw.item()
        point_spread_predicted = point_spread_pred_raw.item()

        st.subheader('Prediction Results:')
        st.write(f"**Predicted Home Win Probability:** {home_win_probability:.2f}")
        st.write(f"**Predicted Outcome:** {home_win_predicted}")
        st.write(f"**Predicted Total Points:** {total_points_predicted:.2f}")
        st.write(f"**Predicted Point Spread (Home - Away):** {point_spread_predicted:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are valid and try again.")

st.markdown("""
--- 
\n**Note**: This is a demonstration. For real-world use, more robust data validation and error handling would be necessary. 
Team and player statistics should ideally be fetched from a live data source.
""")
