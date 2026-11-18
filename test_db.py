
import streamlit as st
from sqlalchemy import text
from db import SessionLocal

st.title("MySQL Connection Test")

try:
    with SessionLocal() as session:
        database_name = session.execute(
            text("SELECT DATABASE()")
        ).scalar_one()

        mysql_version = session.execute(
            text("SELECT VERSION()")
        ).scalar_one()

    st.success("Connection successful!")
    st.write("Database:", database_name)
    st.write("MySQL version:", mysql_version)

except Exception as e:
    st.error("Connection failed.")
    st.exception(e)
