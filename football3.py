import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import unicodedata
import matplotlib.pyplot as plt
from mplsoccer import PyPizza, FontManager

# Load the trained scaler and model
preprocessor = joblib.load('scaler1.pkl')
model = joblib.load('xgboost.pkl')

@st.cache_data  # Cache the data to avoid frequent reloading
def load_data():
    # Load data directly from the raw GitHub link
    url = "https://raw.githubusercontent.com/YonaskT/footy/main/merged_players.csv"
    try:
        df = pd.read_csv(url)
        st.write("Data Loaded Successfully!")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

# Load data into the app
df = load_data()

# Normalize accented characters in unique_id and player_id
df['unique_id'] = df['unique_id'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)
df['player_id'] = df['player_id'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() if isinstance(x, str) else x)

# Define features for similarity calculation
features = [
    'gls', 'ast', 'xg', 'xag', 'min', 'mp', 'npxg',
    'touches', 'cmp_pcnt', 'prgp', 'kp', 'prgc', 'prgdist', 'ppa',
    'pass_into_final_third', 'carries_into_final_third', 'carries',
    'tkl_plus_int', 'blocks', 'tkl_pcnt', 'clr',
    'sot_pcnt', 'sh_per_90', 'sot_per_90'
]

# Create feature matrix
feature_matrix = df[features].fillna(0)

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
        idx = df[df['player_x'] == player_name].index[0]
    except IndexError:
        st.write(f"Player {player_name} not found in database")
        return pd.DataFrame()
    
    # Get similarity scores for this player
    player_similarities = similarity_matrix[idx]
    
    # Get indices of most similar players (excluding self)
    similar_indices = player_similarities.argsort()[::-1][1:n+1]
    
    similar_players = df.iloc[similar_indices]
    similar_players['similarity_score'] = player_similarities[similar_indices]
    
    return similar_players

# Define function to compute percentiles within each position group
def compute_position_percentiles(df, main_position):
    pos_group = df[df['main_position'] == main_position]  # Filter by position
    
    for col in [
        'gls', 'ast', 'tkl_pcnt', 'xg', 'prgc', 'kp', 
        'ppa', 'prgp', 'cmp_pcnt', 'sot_per_90'
    ]:
        percentile_col = f"{col}_percentile"
        df.loc[df['main_position'] == main_position, percentile_col] = pos_group[col].rank(pct=True) * 100

    return df

# Apply percentile calculation by position
for pos in df['main_position'].unique():
    df = compute_position_percentiles(df, pos)

# Streamlit app
st.title('Player Analysis and Market Value Prediction')

# Sidebar for user input
st.sidebar.header('Player Selection')
selected_player = st.sidebar.selectbox('Select a player', df['player_x'].unique())

# Predict the market value for the selected player
st.header(f"Predicted Market Value for {selected_player}")
try:
    player_data = df[df['player_x'] == selected_player]
    player_processed = preprocessor.transform(player_data)
    predicted_value = model.predict(player_processed)[0]
    st.success(f"**Predicted Market Value:** €{predicted_value:,.2f}")
except Exception as e:
    st.error(f"Error during prediction: {e}")

# Show similar players
similar_players = get_player_recommendations(selected_player)
if not similar_players.empty:
    st.write(f'Most similar players to {selected_player}:')
    st.write(similar_players[['player_x', 'squad', 'main_position', 'similarity_score']])

# Select a player for pizza plot
player_data = df[df['player_x'] == selected_player].iloc[0]

# Define parameters and values
params = [
    'Goals','Assist','Tackle Percentage','Expected Goals','Progressive Carries', 
    'Key Passes', 'Passes into Penalty Area', 'Progressive Passes', 'Pass Completion %', 
    'Shots on Target per 90'
]
percentile_columns = [
    'gls_percentile', 'ast_percentile', 'tkl_pcnt_percentile', 
    'xg_percentile', 'prgc_percentile', 'kp_percentile', 
    'ppa_percentile', 'prgp_percentile', 'cmp_pcnt_percentile',  'sot_per_90_percentile'
]

# Fetch percentile values for the selected player
values = [round(player_data[percentile_column], 2) for percentile_column in percentile_columns]

# Fonts
font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf')
font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Italic.ttf')
font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf')

# Slice colors (grouped by attack, midfield, defense)
slice_colors = ["#FF6347"] * 4 + ["#4682B4"] * 4 + ["#32CD32"] * 2
text_colors = ["#FFFFFF"] * 10

# Instantiate PyPizza class
baker = PyPizza(
    params=params,                  
    background_color="#222222",     
    straight_line_color="#000000",  
    straight_line_lw=1,             
    last_circle_color="#000000",    
    last_circle_lw=1,               
    other_circle_lw=0,              
    inner_circle_size=20            
)

# Plot pizza
fig, ax = baker.make_pizza(
    values,                          
    figsize=(10, 8.5),                
    color_blank_space="same",        
    slice_colors=slice_colors,       
    value_colors=text_colors,        
    value_bck_colors=slice_colors,   
    blank_alpha=0.4,                 
    kwargs_slices=dict(
        edgecolor="#000000", zorder=2, linewidth=1
    ),                               
    kwargs_params=dict(
        color="#F2F2F2", fontsize=11,
        fontproperties=font_normal.prop, va="center"
    ),                               
    kwargs_values=dict(
        color="#F2F2F2", fontsize=11,
        fontproperties=font_normal.prop, zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                             
)

# Add title
fig.text(
    0.515, 0.975, f"{selected_player} ({player_data['main_position']})", size=14,
    ha="center", fontproperties=font_bold.prop, color="#F2F2F2"
)
# Add subtitle
fig.text(
    0.515, 0.955,
    "Position-Based Percentiles | Season 2024-25",
    size=11,
    ha="center", fontproperties=font_bold.prop, color="#F2F2F2"
)
# Add credits
CREDIT_1 = "data: FBref"

fig.text(
    0.99, 0.02, f"{CREDIT_1}", size=9,
    fontproperties=font_italic.prop, color="#F2F2F2",
    ha="right"
)

st.pyplot(fig)

# Additional Visuals
st.header("Player Comparison and Insights")



# Player Comparison
st.sidebar.header('Player Comparison')
player1 = st.sidebar.selectbox('Select first player', df['player_x'].unique(), key='player1')
player2 = st.sidebar.selectbox('Select second player', df['player_x'].unique(), key='player2')

def create_radar_plot(players, metrics):
    # Get data for selected players and metrics
    plot_data = df[df['player_x'].isin(players)][['player_x'] + metrics].set_index('player_x')
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    for idx, player in enumerate(players):
        values = plot_data.loc[player].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=player)
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Player Comparison Radar Plot")
    
    return fig

if player1 and player2:
    st.write(f"Comparing {player1} and {player2}")
    
    # Select metrics for comparison
    comparison_metrics = ['gls', 'ast', 'xg', 'npxg', 'xag', 'sh_per_90', 'sot_per_90']
    
    # Create and display the radar plot
    radar_plot = create_radar_plot([player1, player2], comparison_metrics)
    st.pyplot(radar_plot)