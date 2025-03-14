import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import unicodedata
import psycopg2
import joblib
import python-dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(5432)
}
# Create a connection string for SQLAlchemy
connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
engine = create_engine(connection_string)

# Load data
@st.cache_data
def load_data():
    final_merged_df = pd.read_sql_query("""select * from prediction""", con=engine)
    return final_merged_df

final_merged_df = load_data()

# Normalize accented characters in unique_id and player_id
final_merged_df['unique_id'] = final_merged_df['unique_id'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)
final_merged_df['player_id'] = final_merged_df['player_id'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)

# Define features for similarity calculation
features = [
    # Performance metrics
    'gls', 'ast', 'xg', 'xag', 'min', 'mp', 'npxg',
    
    # Possession/Passing
    'touches', 'cmp_pcnt', 'prgp', 'kp', 'prgc', 'prgdist', 'ppa',
    'pass_into_final_third', 'carries_into_final_third', 'carries',
    
    # Defensive actions 
    'tkl_plus_int', 'blocks', 'tkl_pcnt', 'clr',
    
    # Shooting
    'sot_pcnt', 'sh_per_90', 'sot_per_90'
]

# Create feature matrix
feature_matrix = final_merged_df[features].fillna(0)

# Normalize features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(scaled_features)

def get_player_recommendations(player_name, n=5):
    """
    Get n most similar players to the given player name
    """
    # Get player index
    try:
        idx = final_merged_df[final_merged_df['player_x'] == player_name].index[0]
    except IndexError:
        st.write(f"Player {player_name} not found in database")
        return pd.DataFrame()
    
    # Get similarity scores for this player
    player_similarities = similarity_matrix[idx]
    
    # Get indices of most similar players (excluding self)
    similar_indices = player_similarities.argsort()[::-1][1:n+1]
    
    similar_players = final_merged_df.iloc[similar_indices]
    similar_players['similarity_score'] = player_similarities[similar_indices]
    
    return similar_players

def predict_player_value(player_name):
    """
    Predict the market value of the given player
    """
    # Select numeric columns for prediction
    numeric_features = final_merged_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('value')  # Remove the target column

    # Prepare the data
    X = final_merged_df[numeric_features]
    y = final_merged_df['value']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the value for the selected player
    player_data = final_merged_df[final_merged_df['player_x'] == player_name][numeric_features]
    predicted_value = model.predict(player_data)[0]

    return predicted_value

# Title and introductory text
st.title('Player Analysis App')
st.write('Select a player from the sidebar to see similar players and predict their market value.')

# Sidebar for user input
st.sidebar.header('Player Selection')
selected_player = st.sidebar.selectbox('Select a player', final_merged_df['player_x'].unique())

# Show similar players
similar_players = get_player_recommendations(selected_player)
if not similar_players.empty:
    st.write(f'Most similar players to {selected_player}:')
    st.write(similar_players[['player_x', 'squad', 'main_position', 'similarity_score']])

# Predict player value
predicted_value = predict_player_value(selected_player)
st.write(f'Predicted market value for {selected_player}: ${predicted_value:,.2f}')

# Run the app
if __name__ == '__main__':
    st.write('Select a player from the sidebar to see similar players and predict their market value.')
