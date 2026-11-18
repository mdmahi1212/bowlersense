"""Alembic environment. Uses the engine passed in ``config.attributes`` when present."""

from __future__ import annotations

from alembic import context
from sqlalchemy import Engine, create_engine

from bowlsense.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    engine = config.attributes.get("engine")
    if not isinstance(engine, Engine):
        url = config.get_main_option("sqlalchemy.url")
        if not url:
            raise RuntimeError("No database engine or sqlalchemy.url configured for Alembic.")
        engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
