import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    conn.execute(text("ALTER ROLE authenticator SET pgrst.db_schemas = 'public, graphql_public, horsebet';"))
    conn.execute(text("NOTIFY pgrst, 'reload config';"))
    conn.commit()
print("Success")
