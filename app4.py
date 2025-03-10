import streamlit as st
import pandas as pd
import joblib
import os

# Set page title
st.set_page_config(page_title="Player Value Predictor", layout="wide")

# Create a function to load model files
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('scaler.pkl')
        model = joblib.load('best_xgboost.pkl')
        return preprocessor, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Create a function to load data
@st.cache_data
def load_data():
    # Option 1: Use a CSV file stored in the repository
    try:
        return pd.read_csv('player_data.csv')
    except FileNotFoundError:
        # Option 2: Use Streamlit secrets for database connection
        try:
            import psycopg2
            from sqlalchemy import create_engine
            
            # Get credentials from Streamlit secrets
            db_credentials = st.secrets["postgres"]
            
            # Create connection string
            connection_string = f"postgresql+psycopg2://{db_credentials['user']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['dbname']}"
            engine = create_engine(connection_string)
            
            # Load data from database
            df = pd.read_sql_query("""select * from prediction""", con=engine)
            return df
        except Exception as e:
            st.error(f"Error connecting to database: {e}")
            # Provide sample data as fallback
            return pd.DataFrame({
                'player_x': ['Sample Player 1', 'Sample Player 2'],
                # Add other columns your model expects
            })

# Main app
st.title("Player Market Value Predictor")

# Load models
preprocessor, model = load_models()

# Check if models loaded successfully
if preprocessor is None or model is None:
    st.error("Failed to load models. Please check that 'scaler.pkl' and 'best_xgboost.pkl' are in the repository.")
    st.stop()

# Load data
with st.spinner("Loading player data..."):
    df = load_data()

# Display info about the app
with st.expander("About this app"):
    st.write("""
    This app predicts the market value of players based on various metrics.
    Select a player from the dropdown and click 'Predict' to see their estimated market value.
    """)

# Sidebar for user input
st.sidebar.header('Player Selection')
if not df.empty and 'player_x' in df.columns:
    selected_player = st.sidebar.selectbox('Select a player', df['player_x'].unique())
    
    # Add a predict button
    if st.sidebar.button('Predict', type="primary"):
        st.subheader(f"Prediction for {selected_player}")
        
        # Progress bar for visual feedback
        progress_bar = st.progress(0)
        
        try:
            # Get player data
            player_data = df[df['player_x'] == selected_player]
            
            if player_data.empty:
                st.warning(f"No data found for {selected_player}")
            else:
                # Update progress
                progress_bar.progress(25)
                
                # Check for required columns
                required_columns = preprocessor.feature_names_in_
                missing_columns = [col for col in required_columns if col not in player_data.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Prepare data for preprocessing
                    player_data = player_data[required_columns]
                    progress_bar.progress(50)
                    
                    # Transform the data
                    player_processed = preprocessor.transform(player_data)
                    progress_bar.progress(75)
                    
                    # Make prediction
                    predicted_value = model.predict(player_processed)[0]
                    progress_bar.progress(100)
                    
                    # Display result
                    st.success(f'Predicted market value for {selected_player}: ${predicted_value:,.2f}')
                    
                    # Show player stats
                    st.subheader("Player Statistics")
                    st.dataframe(player_data)
                    
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("No player data available or incorrect data format")

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("Made with Streamlit and ❤️")
