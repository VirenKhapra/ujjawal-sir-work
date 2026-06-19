import asyncio
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.engine import make_url


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", identifier):
        raise ValueError(f"Unsafe PostgreSQL identifier: {identifier}")
    return f'"{identifier}"'


async def _create_database(admin_url: str, database_name: str) -> str:
    admin = make_url(admin_url)
    test_url = admin.set(database=database_name)
    conn = await asyncpg.connect(dsn=_asyncpg_dsn(str(admin)))
    try:
        await conn.execute(f"CREATE DATABASE {_quote_identifier(database_name)}")
    finally:
        await conn.close()
    return str(test_url)


async def _drop_database(admin_url: str, database_name: str) -> None:
    admin = make_url(admin_url)
    conn = await asyncpg.connect(dsn=_asyncpg_dsn(str(admin)))
    try:
        quoted = _quote_identifier(database_name)
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1",
            database_name,
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {quoted}")
    finally:
        await conn.close()


async def _fetch_callback_schema(database_url: str) -> dict[str, set[str]]:
    conn = await asyncpg.connect(dsn=_asyncpg_dsn(database_url))
    try:
        tables = {
            row["table_name"]
            for row in await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            )
        }
        columns = {
            (row["table_name"], row["column_name"], row["is_nullable"])
            for row in await conn.fetch(
                """
                SELECT table_name, column_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                """
            )
        }
        constraints = {
            row["constraint_name"]
            for row in await conn.fetch(
                """
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_schema = 'public'
                """
            )
        }
    finally:
        await conn.close()
    return {"tables": tables, "columns": columns, "constraints": constraints}


def test_callback_persistence_migrations_match_postgresql_schema():
    admin_url = os.getenv("POSTGRES_ADMIN_DATABASE_URL")
    if not admin_url:
        pytest.skip("Set POSTGRES_ADMIN_DATABASE_URL to run PostgreSQL migration parity test.")

    database_name = f"finflow_migration_{uuid.uuid4().hex[:12]}"
    backend_dir = Path(__file__).resolve().parents[1]
    database_url = asyncio.run(_create_database(admin_url, database_name))
    try:
        env = {**os.environ, "DATABASE_URL": database_url}
        subprocess.run(
            [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"],
            cwd=backend_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        schema = asyncio.run(_fetch_callback_schema(database_url))
    finally:
        asyncio.run(_drop_database(admin_url, database_name))

    assert {"needs_review_jobs", "dead_letter_jobs", "callback_events"} <= schema["tables"]
    assert ("needs_review_jobs", "source_event_id", "YES") in schema["columns"]
    assert ("dead_letter_jobs", "source_event_id", "YES") in schema["columns"]
    assert ("dead_letter_jobs", "created_at", "NO") in schema["columns"]
    assert ("callback_events", "event_id", "NO") in schema["columns"]
    assert ("callback_events", "received_at", "NO") in schema["columns"]
    assert "uq_needs_review_jobs_source_event_id" in schema["constraints"]
    assert "uq_dead_letter_jobs_source_event_id" in schema["constraints"]
    assert "uq_callback_events_event_id" in schema["constraints"]
