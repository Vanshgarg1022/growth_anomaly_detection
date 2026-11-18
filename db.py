
import streamlit as st
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker


@st.cache_resource
def get_session_factory():
    cfg = st.secrets["database"]

    url = URL.create(
        "mysql+mysqlconnector",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["database"],
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )


SessionLocal = get_session_factory()
