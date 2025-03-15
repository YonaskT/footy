import os
import time
import requests
import numpy as np
import pandas as pd
import schedule
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Column Handling
def rename_columns_for_sql(df):
    special_char_mapping = {'+': '_plus_', '-': '_minus_', '/': '_per_', ' ': '_', '%': '_pcnt', '90s': 'nineties'}
    df.columns = df.columns.astype(str)
    df.columns = [col.lower() for col in df.columns]
    for char, replacement in special_char_mapping.items():
        df.columns = [col.replace(char, replacement) for col in df.columns]
    return df

def convert_age_to_birthdate(age_str):
    if pd.isna(age_str):
        return np.nan
    try:
        years, days = map(int, age_str.split('-'))
        now = datetime.now()
        birth_date = now - timedelta(days=days + 1)
        birth_date = birth_date.replace(year=birth_date.year - years)
        return birth_date.strftime('%y%m%d')
    except Exception:
        return np.nan

def generate_unique_id(player_name, birth_date):
    if pd.isna(player_name) or pd.isna(birth_date):
        return np.nan
    try:
        last_name = player_name.split()[-1][:5].lower()
        return f"{last_name}_{birth_date}"
    except Exception:
        return np.nan

def process_table(df, drop_columns, string_cols, float_cols, rename_dict=None):
    df = df.drop(columns=drop_columns, errors='ignore')
    if 'age' in df.columns:
        df['birth_date'] = df['age'].apply(convert_age_to_birthdate)
        df = df.drop(columns=['age'])
    if rename_dict:
        df = df.rename(columns=rename_dict)
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    df = df[df['player'] != 'Player']
    df['unique_id'] = df.apply(lambda row: generate_unique_id(row['player'], row['birth_date']), axis=1)
    return df

def extract_fbref_data():
    urls = {
        'shooting': ('https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats', 'stats_shooting'),
        'passing': ('https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats', 'stats_passing'),
        'defense': ('https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats', 'stats_defense'),
        'standard': ('https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats', 'stats_standard'),
        'possession': ('https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats', 'stats_possession')
    }
    dataframes = {}
    for key, (url, table_id) in urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            html = response.text.replace('<!--', '').replace('-->', '')
            df = pd.read_html(html, attrs={'id': table_id})[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel()
            df = df.loc[:, ~df.columns.duplicated()]
            df = rename_columns_for_sql(df)
            dataframes[key] = df
            print(f"✅ Successfully fetched and processed {key} data.")
        except Exception as e:
            print(f"❌ Error fetching {key} data: {e}")
    return dataframes

def save_to_supabase(data, table_name):
    try:
        data_dict = data.to_dict(orient='records')
        response = supabase.table(table_name).upsert(data_dict).execute()
        print(f"✅ Successfully saved {table_name} data to Supabase.")
    except Exception as e:
        print(f"❌ Error saving {table_name} data to Supabase: {e}")

def job():
    print("\n🚀 Running scheduled scraping and database update...")
    data = extract_fbref_data()
    processed_tables = {}
    
    table_processors = {
        'standard': lambda df: process_table(df, ['rk', 'nation', 'pos', 'born'], ['player', 'squad', 'comp'], ['nineties', 'xg', 'npxg']),
        'defense': lambda df: process_table(df, ['rk', 'nation', 'pos', 'born'], ['player', 'squad', 'comp'], ['nineties', 'tkl_pcnt']),
        'passing': lambda df: process_table(df, ['rk', 'nation', 'pos', 'born'], ['player', 'squad', 'comp'], ['nineties', 'cmp_pcnt'], {'1_per_3': 'pass_into_final_third'}),
        'shooting': lambda df: process_table(df, ['rk', 'nation', 'pos', 'born'], ['player', 'squad', 'comp'], ['nineties', 'sot_pcnt', 'sh_per_90']),
        'possession': lambda df: process_table(df, ['rk', 'nation', 'pos', 'born'], ['player', 'squad', 'comp'], ['nineties', 'succ_pcnt'], {'1_per_3': 'carries_into_final_third'})
    }
    
    for key, process_func in table_processors.items():
        if key in data:
            processed_tables[key] = process_func(data[key])
            save_to_supabase(processed_tables[key], key)
    print("✅ Job completed successfully!")

# Schedule the job to run at 08:00 AM daily
schedule.every().day.at("01:37").do(job)

print("⏳ Scheduler started. Waiting for the next run ...")
while True:
    schedule.run_pending()
    time.sleep(60)
