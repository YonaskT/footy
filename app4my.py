import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    'dbname': os.getenv('MY_DB_NAME'),
    'user': os.getenv('MY_DB_USER'),
    'password': os.getenv('MY_DB_PASSWORD'),
    'host': os.getenv('MY_DB_HOST'),
    'port': int(os.getenv('MY_DB_PORT'))
}

# Create a connection string for SQLAlchemy
connection_string = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
engine = create_engine(connection_string)

# Load the trained scaler and model
preprocessor = joblib.load('scaler.pkl')
model = joblib.load('best_xgboost.pkl')

# Load the data
@st.cache_data
def load_data():
    df = pd.read_sql_query("""select * from prediction""", con=engine)
    return df

df = load_data()

# Debugging: Display the loaded data
st.write("Loaded Data:", df.sample(5))

# Sidebar for user input
st.sidebar.header('Player Selection')
selected_player = st.sidebar.selectbox('Select a player', df['player_x'].unique())

# Add a predict button
if st.button('Predict'):
    # Debugging: Display selected player
    st.write("Selected Player:", selected_player)
    
    # Preprocess the data
    try:
        preprocessed_data = preprocessor.transform(df)
        st.write("Preprocessed Data Shape:", preprocessed_data.shape)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
    
    # Predict the value for the selected player
    try:
        player_data = df[df['player_x'] == selected_player]
        st.write("Player Data:", player_data)
        
        player_processed = preprocessor.transform(player_data)
        st.write("Player Processed Data Shape:", player_processed.shape)
        
        predicted_value = model.predict(player_processed)[0]
        st.write(f'Predicted market value for {selected_player}: ${predicted_value:,.2f}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    