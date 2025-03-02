import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import PyPizza, FontManager

# Load data
df=pd.read_csv('prediction.csv')

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
st.title("Player Performance Radar Chart")

# Select a player
player_name = st.selectbox('Select a player:', df['player_x'].unique())
player_data = df[df['player_x'] == player_name].iloc[0]

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
slice_colors = ["#1A78CF"] * 4 + ["#FF9300"] * 4 + ["#D70232"] * 2
text_colors = ["#000000"] * 9 + ["#F2F2F2"] * 1

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
    0.515, 0.975, f"{player_name} ({player_data['main_position']})", size=16,
    ha="center", fontproperties=font_bold.prop, color="#F2F2F2"
)
# Add subtitle
fig.text(
    0.515, 0.955,
    "Position-Based Percentiles | Season 2024-25",
    size=13,
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