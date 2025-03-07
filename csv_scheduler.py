import time
import schedule
import pandas as pd
import psycopg2
import os
import logging
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', '5432')
}

# Verify that SCHEDULE_TIME is loaded correctly
schedule_time = os.getenv('SCHEDULE_TIME', '08:15')
logging.info(f"SCHEDULE_TIME loaded: {schedule_time}")

# Function to update the CSV file
def update_csv():
    try:
        # Connect to the database
        with psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        ) as conn:
            logging.info("Connected to the database successfully!")

            # Load data into a DataFrame
            query = "SELECT * FROM merged_players;"
            df = pd.read_sql(query, conn)

            # Save DataFrame to CSV
            df.to_csv('merged_players.csv', index=False)
            logging.info("CSV updated successfully!")

    except Exception as e:
        logging.error(f"Error updating CSV: {e}")

# Schedule the task to run every day at the specified time
logging.info(f"Scheduling task to run every day at {schedule_time}")
schedule.every().day.at(schedule_time).do(update_csv)

# Keep the script running
while True:
    schedule.run_pending()
    logging.info("Scheduler is running...")
    time.sleep(60)



