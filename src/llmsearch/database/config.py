from collections import namedtuple

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DBSettings = namedtuple("DBSettings", "SessionLocal engine")


def get_local_session(db_path: str) -> DBSettings:
    sql_alchemy_database_url = f"sqlite:///{db_path}"
    engine = create_engine(
        sql_alchemy_database_url, connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return DBSettings(SessionLocal=SessionLocal, engine=engine)


Base = declarative_base()
