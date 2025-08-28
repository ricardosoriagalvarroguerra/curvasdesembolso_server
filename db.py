import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


load_dotenv()


@lru_cache(maxsize=1)
def get_database_url() -> str:
	url = os.getenv("DATABASE_URL")
	if not url:
		# Allow missing URL in test contexts; some endpoints/tests may not need DB.
		# We avoid raising here; downstream callers should handle absent URL.
		return ""
	return url


@lru_cache(maxsize=1)
def get_engine():
	url = get_database_url()
	if not url:
		# Defer engine creation if URL missing
		raise RuntimeError("DATABASE_URL no configurado. Define la variable de entorno o el archivo .env")
	engine = create_engine(
		url,
		pool_pre_ping=True,
		pool_size=10,
		max_overflow=10,
		connect_args={
			"options": "-c statement_timeout=10000 -c idle_in_transaction_session_timeout=10000",
		},
	)
	return engine


@lru_cache(maxsize=1)
def get_session_maker():
	engine = get_engine()
	return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
	"""FastAPI dependency that yields a SQLAlchemy session."""
	try:
		SessionLocal = get_session_maker()
	except Exception:
		# If DB is not configured (e.g., tests monkeypatch data access), yield None.
		yield None
		return
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


