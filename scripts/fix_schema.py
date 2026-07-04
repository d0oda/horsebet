import sqlite3
import os
import subprocess
import re

DB_PATH = 'horsebet.db'
BACKUP_PATH = 'horsebet_corrupt.db'
DUMP_PATH = 'clean_dump.sql'

TABLES = [
    'courses', 'trainers', 'jockeys', 'horses', 'races', 
    'entries', 'results', 'odds_snapshots', 'predictions', 'value_bets'
]

def fix_schema():
    print("1. Connecting to database to purge NULL ids...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_deleted = 0
    for table in TABLES:
        cursor.execute(f"DELETE FROM {table} WHERE id IS NULL")
        deleted = cursor.rowcount
        total_deleted += deleted
        print(f"  - Deleted {deleted} corrupted rows from {table}")
    
    conn.commit()
    conn.close()
    print(f"Total rows purged: {total_deleted}")

    print("\n2. Dumping cleaned database to SQL file...")
    with open(DUMP_PATH, 'w') as f:
        subprocess.run(['sqlite3', DB_PATH, '.dump'], stdout=f, check=True)
        
    print("\n3. Modifying schema in dump file to use INTEGER PRIMARY KEY AUTOINCREMENT...")
    with open(DUMP_PATH, 'r') as f:
        sql_content = f.read()
    
    # Replace 'id BIGINT,' with 'id INTEGER PRIMARY KEY AUTOINCREMENT,'
    # We use regex to handle any spacing
    modified_sql = re.sub(
        r'\bid\s+BIGINT\s*,', 
        'id INTEGER PRIMARY KEY AUTOINCREMENT,', 
        sql_content, 
        flags=re.IGNORECASE
    )
    
    with open(DUMP_PATH, 'w') as f:
        f.write(modified_sql)
        
    print("\n4. Backing up old database and creating new one...")
    os.rename(DB_PATH, BACKUP_PATH)
    
    with open(DUMP_PATH, 'r') as f:
        subprocess.run(['sqlite3', DB_PATH], stdin=f, check=True)
        
    print(f"\n5. Done! New schema applied to {DB_PATH}.")
    print(f"Old database backed up to {BACKUP_PATH}.")
    
if __name__ == '__main__':
    fix_schema()
