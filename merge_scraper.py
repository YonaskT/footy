import time
import schedule
import pandas as pd
import numpy as np
import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
engine = create_engine(connection_string)

import time
import schedule
import pandas as pd
import numpy as np
import os
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
engine = create_engine(connection_string)

# SQL query to update merged_players
UPDATE_QUERY = """
DROP TABLE IF EXISTS merged_players;
CREATE TABLE merged_players AS
SELECT 
    p.unique_id AS unique_id,
    p.player AS player_x,
    p.mp, p.starts, p.min, p.nineties, p.gls, p.ast, p.g_plus_a, p.g_minus_pk, 
    p.pk, p.pkatt, p.crdy, p.crdr, p.xg, p.npxg, p.xag, p.npxg_plus_xag, 
    p.prgc, p.prgp, p.prgr, p.g_plus_a_minus_pk, p.xg_plus_xag, p.matches, 
    p.squad, p.comp,
    COALESCE(TO_DATE(p.birth_date::TEXT, 'YYMMDD'), p.birth_date::DATE) AS birth_date,
    p.tkl_pcnt, p.tkl, p.tklw, p.def_3rd, p.mid_3rd, p.att_3rd, p.att, p.lost, 
    p.blocks, p.pass, p.int, p.tkl_plus_int, p.clr, p.err, p.sot_pcnt, 
    p.sh_per_90, p.sot_per_90, p.g_per_sh, p.g_per_sot, p.dist, 
    p.fk, p.npxg_per_sh, p.g_minus_xg, p.cmp_pcnt, p.cmp, 
    p.totdist, p.prgdist, p.xa, p.a_minus_xag, p.kp, p.pass_into_final_third, 
    p.ppa, p.crspa, p.succ_pcnt, p.tkld_pcnt, p.touches, p.def_pen, p.att_pen, 
    p.live, p.succ, p.tkld, p.carries, p.carries_into_final_third, p.cpa, p.mis, 
    p.dis, p.rec,
    COALESCE(m.player_id, m2.player_id) AS player_id,
    COALESCE(m.player, m2.player) AS player_y,
    COALESCE(m.contract_expiry, m2.contract_expiry) AS contract_expiry,
    COALESCE(m.main_position, m2.main_position) AS main_position,
    COALESCE(m.value, m2.value) AS value,
    COALESCE(m.height, m2.height) AS height,
    COALESCE(m.current_club, m2.current_club) AS current_club,
    COALESCE(m.league, m2.league) AS league,
    COALESCE(TO_DATE(m.date_of_birth::TEXT, 'YYYY-MM-DD'), m2.date_of_birth::DATE) AS date_of_birth,
    COALESCE(m.days_until_expiry, m2.days_until_expiry) AS days_until_expiry,
    EXTRACT(YEAR FROM AGE(COALESCE(
        COALESCE(TO_DATE(m.date_of_birth::TEXT, 'YYYY-MM-DD'), m2.date_of_birth::DATE),
        COALESCE(TO_DATE(p.birth_date::TEXT, 'YYMMDD'), p.birth_date::DATE)
    ))) AS age,
    CASE 
        WHEN m.player_id IS NOT NULL THEN 'id_match'
        WHEN m2.player_id IS NOT NULL THEN 'birthdate_name_match'
        ELSE 'no_match'
    END AS match_type
FROM performance p
LEFT JOIN market m
    ON UNACCENT(LOWER(p.unique_id)) = UNACCENT(LOWER(m.player_id))
INNER JOIN market m2
    ON m.player_id IS NULL
    AND ABS(CAST(
        COALESCE(TO_DATE(p.birth_date::TEXT, 'YYMMDD'), p.birth_date::DATE) - 
        COALESCE(TO_DATE(m2.date_of_birth::TEXT, 'YYYY-MM-DD'), m2.date_of_birth::DATE) AS INTEGER
    )) < 2
    AND (UNACCENT(LOWER(p.player)) LIKE '%' || UNACCENT(LOWER(m2.player)) || '%'
        OR UNACCENT(LOWER(m2.player)) LIKE '%' || UNACCENT(LOWER(p.player)) || '%');
"""

def update_merged_players():
    try:
        logging.info("Job started: Updating merged_players table")
        start_time = time.time()
        
        with engine.begin() as connection:
            # Split the query into individual statements
            statements = UPDATE_QUERY.split(';')
            for statement in statements:
                if statement.strip():
                    connection.execute(text(statement))
        
        end_time = time.time()
        duration = end_time - start_time
        
        logging.info(f"Job completed: merged_players table updated successfully")
        logging.info(f"Job duration: {duration:.2f} seconds")
    except Exception as e:
        logging.error(f"Error updating merged_players: {e}")
        # Optional: Log the full traceback for more detailed debugging
        import traceback
        logging.error(traceback.format_exc())

def main():
    # Log scheduler start
    logging.info("Scheduler started")
    logging.info(f"Next run scheduled at 08:05 AM daily")

    # Schedule the update 
    schedule.every().day.at("08:05").do(update_merged_players)


    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error in scheduler: {e}")

if __name__ == "__main__":
    main()
