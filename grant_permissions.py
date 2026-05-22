import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    conn.execute(text("GRANT USAGE ON SCHEMA horsebet TO anon, authenticated;"))
    conn.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA horsebet TO anon, authenticated;"))
    conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA horsebet GRANT SELECT ON TABLES TO anon, authenticated;"))
    conn.commit()
print("Granted permissions")
