import os

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get(
    "WARDEN_DATABASE_URL",
    "postgresql://warden:warden@localhost:5432/warden",
)

# Opened/closed by the app lifespan in app.py, not at import time.
pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10, open=False)


def jsonb(value):
    # Store missing values as SQL NULL, not JSON null.
    return None if value is None else Jsonb(value)
