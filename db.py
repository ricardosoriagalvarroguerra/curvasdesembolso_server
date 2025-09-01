import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


load_dotenv()


@lru_cache(maxsize=1)
def get_database_url() -> str:
        """Return a PostgreSQL connection URL from the environment.

        Railway exposes both ``DATABASE_URL`` (internal) and
        ``DATABASE_PUBLIC_URL`` (external).  For local usage we also allow
        constructing the URL from ``POSTGRES_*`` variables.  Tests that do not
        require a database can run without any of these variables defined.
        """

        url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL")

        if not url:
                host = os.getenv("POSTGRES_HOST")
                db = os.getenv("POSTGRES_DB")
                user = os.getenv("POSTGRES_USER", "postgres")
                password = os.getenv("POSTGRES_PASSWORD")
                port = os.getenv("POSTGRES_PORT", "5432")
                if host and db:
                        if password:
                                url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
                        else:
                                url = f"postgresql://{user}@{host}:{port}/{db}"

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


