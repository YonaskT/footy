# Player Analysis and Market Value Prediction and Web App
This project is designed to provide detailed analysis and market value predictions for football players. 
The [app](https://recommenderfootball.streamlit.app/) leverages machine learning models and data visualization techniques to offer insights into player performance and similarity comparisons.

The main aim of this project is to create a comprehensive dataset for analyzing football player performances and predicting market value, leveraging data from multiple sources and perform extensive data cleaning, preprocessing analyzing and getting insights.

## Project Overview

- Build player scouting recommender system
- understand which variables increase the market value per player
- Predicts player market value based on their performance metrics
- Provides insights and recommendations for clubs, scouts
- identify over/under performing players based on their market value
- identify undervalued or overvalued players

### Data Source:

- performance data is scraped from [fbref](https://fbref.com/en/)
- market value and other player rofile data is scraped from [transfermarkt](https://www.transfermarkt.com/)
- The data is scraped from the above two football statistical websites and updated everyday using a scheduler and stored in a database
- The app retrieves player data from a PostgreSQL database

### Key Features of the app:

1. Player Selection:

Users can select a player from a dropdown menu in the sidebar.
The app displays the most similar players based on performance metrics.

2. Player Similarity:

The app calculates and displays the top 5 players most similar to the selected player using cosine similarity on various performance metrics.

3. Performance Visualization:

A radar chart (pizza plot) is generated to visualize the selected player's performance across multiple metrics.
The chart includes parameters such as goals, assists, tackle percentage, expected goals, progressive carries, key passes, passes into the penalty area, progressive passes, pass completion percentage, and shots on target per 90 minutes.

4. Market Value Prediction:

Users can predict the market value of the selected player by clicking the "Predict Market Value" button.
The app uses a pre-trained machine learning model to estimate the player's market value based on their performance data.

### Technical Details:

Data Source: The app retrieves player data from a PostgreSQL database .

Machine Learning: The app uses a pre-trained XGBoost model for market value prediction.

Data Processing: The app employs a combination of StandardScaler for feature scaling.

Visualization: The radar chart is created using the mplsoccer library's PyPizza class.

Framework: The app is built using the Streamlit framework, which allows for interactive and real-time data visualization.

How to Use the app:

Select a Player: Choose a player from the dropdown menu in the sidebar.

View Similar Players: The app will display a list of players most similar to the selected player.

Analyze Performance: A radar chart will be generated to visualize the selected player's performance metrics among players that plays in his position.

Predict Market Value: Click the "Predict Market Value" button to estimate the player's market value.

The [web app](https://recommenderfootball.streamlit.app/) provides a comprehensive tool for football analysts, scouts, and enthusiasts to evaluate player performance and make informed decisions based on data-driven insights.



