import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import unicodedata
# removes warnings
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT'))
}
# Create a connection string for SQLAlchemy
connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
engine=create_engine(connection_string)

# Load data
@st.cache
def load_data():
    # Replace with your actual data loading code
    performance_df = pd.read_sql_query("""select * from performance""",con = engine)
    market_df = pd.read_sql_query("""select * from market""",con = engine)

    return performance_df, market_df

performance_df, market_df = load_data()

# Normalize accented characters in unique_id and player_id
performance_df['unique_id'] = performance_df['unique_id'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)
market_df['player_id'] = market_df['player_id'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)

# Merge data
merged_df = pd.merge(performance_df, market_df, how='inner', left_on='unique_id', right_on='player_id')
final_merged_df = pd.read_sql_query("""select * from prediction""",con = engine)
# Sidebar for user input
st.sidebar.header('Player Selection')
selected_player = st.sidebar.selectbox('Select a player', final_merged_df['player_x'].unique())

# Show similar players
def find_similar_players(player_name, df, n=5):
    player_data = df[df['player_x'] == player_name].drop(columns=['player_x', 'player_y', 'league', 'value'])
    similarities = cosine_similarity(player_data, df.drop(columns=['player_x', 'player_y', 'league', 'value']))
    similar_indices = np.argsort(similarities[0])[::-1][1:n+1]
    return df.iloc[similar_indices]

similar_players = find_similar_players(selected_player, final_merged_df)
st.write('Similar Players:')
st.write(similar_players[['player_x', 'league', 'value']])

# Plot radar plot
def plot_radar(player_name, df):
    player_data = df[df['player_x'] == player_name].drop(columns=['player_x', 'player_y', 'league', 'value']).values.flatten()
    categories = df.drop(columns=['player_x', 'player_y', 'league', 'value']).columns
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=player_data, theta=categories, fill='toself', name=player_name))
    st.plotly_chart(fig)

plot_radar(selected_player, final_merged_df)



# Run the app
if __name__ == '__main__':
    st.title('Player Analysis App')
    st.write('Select a player from the sidebar to see similar players, radar plot, and predicted value.')