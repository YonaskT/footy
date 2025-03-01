import time
import schedule
import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta

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

# Define avg_cols at module level, before the functions
avg_cols = ['tkl_pcnt', 'sot_pcnt', 'sh_per_90', 'sot_per_90', 'g_per_sh', 
            'g_per_sot', 'dist','npxg_per_sh', 'cmp_pcnt', 'succ_pcnt', 'tkld_pcnt']
join_cols = ['squad', 'comp']

def rename_columns_for_sql(df):
    """Renames DataFrame columns for SQL compatibility, ensuring valid column names."""
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
        return birth_date.strftime('%y%m%d')  # YYMMDD format
    except Exception:
        return np.nan

def generate_unique_id(player_name, birth_date):
    if pd.isna(player_name) or pd.isna(birth_date):
        return np.nan
    try:
        last_name = player_name.split()[-1][:5].lower()  # First 5 letters of last name
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
    int_cols = [col for col in df.columns if col not in string_cols + float_cols + ['birth_date']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df = df[df['player'] != 'Player']
    df['unique_id'] = df.apply(lambda row: generate_unique_id(row['player'], row['birth_date']), axis=1)
    return df

def process_standard_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'xg', 'npxg', 'xag', 'npxg_plus_xag', 'g_plus_a_minus_pk', 'xg_plus_xag'])

def process_defense_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born','sh'], ['player', 'squad', 'comp'], ['nineties', 'tkl_pcnt'])

def process_passing_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'cmp_pcnt', 'xag', 'xa', 'a_minus_xag'], {'1_per_3': 'pass_into_final_third'})

def process_shooting_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'sot_pcnt', 'sh_per_90', 'sot_per_90', 'g_per_sh', 'g_per_sot', 'dist', 'xg', 'npxg', 'npxg_per_sh', 'g_minus_xg', 'np:g_minus_xg'])

def process_possession_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'succ_pcnt', 'tkld_pcnt'], {'1_per_3': 'carries_into_final_third'})

def create_performance_table(standard_df, defense_df, shooting_df, passing_df, possession_df):
    """Creates a consolidated performance table from individual stat tables."""
    # List of DataFrames to merge
    dfs = [standard_df, defense_df, shooting_df, passing_df, possession_df]
    
    # Track columns already seen across DataFrames
    seen_cols = set()
    
    # Process each DataFrame
    processed_dfs = []
    for i, (df_name, df) in enumerate(zip(['standard', 'defense', 'shooting', 'passing', 'possession'], dfs)):
        processed_dfs.append(deduplicate_df(df, df_name, seen_cols))
    
    # Merge DataFrames
    performance_df = processed_dfs[0]
    for df_name, df in zip(['defense', 'shooting', 'passing', 'possession'], processed_dfs[1:]):
        performance_df = performance_df.merge(df, on='unique_id', how='inner', suffixes=('', '_dup'))
    
    # Drop duplicated columns
    columns_to_keep = ~performance_df.columns.duplicated()
    performance_df = performance_df.loc[:, columns_to_keep]
    
    return performance_df

def deduplicate_df(df, df_name, seen_cols):
    """Deduplicates DataFrame based on unique_id and removes duplicated features."""
    global avg_cols, join_cols
    
    if df['unique_id'].duplicated().any():
        # Identify numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('unique_id', errors='ignore')
        non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.drop('unique_id', errors='ignore')
        
        # Filter out already seen columns
        new_numeric_cols = [col for col in numeric_cols if col not in seen_cols]
        new_non_numeric_cols = [col for col in non_numeric_cols if col not in seen_cols]
        
        # Create aggregation dictionary
        agg_dict = {}
        for col in new_numeric_cols:
            agg_dict[col] = 'mean' if col in avg_cols else 'sum'
        for col in new_non_numeric_cols:
            agg_dict[col] = (lambda x: ', '.join(x.dropna().unique())) if col in join_cols else 'first'
        
        # Group by unique_id and aggregate
        df_dedup = df[['unique_id'] + list(agg_dict.keys())].groupby('unique_id', as_index=False).agg(agg_dict)
        seen_cols.update(agg_dict.keys())
        return df_dedup
    else:
        new_cols = [col for col in df.columns if col not in seen_cols and col != 'unique_id']
        df_filtered = df[['unique_id'] + new_cols]
        seen_cols.update(new_cols)
        return df_filtered

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

def save_to_postgresql(data, table_name, connection_string):
    try:
        engine = create_engine(connection_string)
        data.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"✅ Successfully saved {table_name} data to PostgreSQL.")
    except Exception as e:
        print(f"❌ Error saving {table_name} data to PostgreSQL: {e}")

def job():
    print("\n🚀 Running scheduled scraping and database update...")
    data = extract_fbref_data()
    connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    
    # Process individual tables
    processed_tables = {}
    table_processors = {
        'standard': process_standard_table,
        'defense': process_defense_table,
        'passing': process_passing_table,
        'shooting': process_shooting_table,
        'possession': process_possession_table
    }
    
    for key, process_func in table_processors.items():
        if key in data:
            processed_tables[key] = process_func(data[key])
            save_to_postgresql(processed_tables[key], key, connection_string)
    
    # Create and save performance table
    if len(processed_tables) == 5:  # Only if all tables were successfully processed
        performance_df = create_performance_table(
            processed_tables['standard'],
            processed_tables['defense'],
            processed_tables['shooting'],
            processed_tables['passing'],
            processed_tables['possession']
        )
        save_to_postgresql(performance_df, 'performance', connection_string)
    
    print("✅ Job completed successfully!")

# Schedule the job to run at 08:00 AM daily
schedule.every().friday.at("02:35").do(job)

print("⏳ Scheduler started. Waiting for the next run ...")

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)  # Wait for 60 seconds before checking again