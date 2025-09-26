import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus

def make_engine(cfg):
    d = cfg["db"]
    if d.get("trusted_connection"):
        cs = (
          f"DRIVER={{{d['driver']}}};SERVER={d['server']};DATABASE={d['database']};"
          "Trusted_Connection=yes;TrustServerCertificate=yes"
        )
    else:
        cs = (
          f"DRIVER={{{d['driver']}}};SERVER={d['server']};DATABASE={d['database']};"
          f"UID={d['username']};PWD={d['password']};TrustServerCertificate=yes"
        )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(cs)}", future=True)
