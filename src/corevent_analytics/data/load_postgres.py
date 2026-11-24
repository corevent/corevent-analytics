from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from corevent_analytics.config import Settings, get_settings
from corevent_analytics.data.generate_synthetic import generate_synthetic_dataset

logger = logging.getLogger(__name__)

TABLE_QUERIES: dict[str, str] = {
    "users": """
        SELECT id, birth_date, created_at
        FROM users
    """,
    "events": """
        SELECT
            id,
            category,
            location_type,
            city_id,
            is_adult_only,
            status,
            start_date,
            end_date,
            created_at
        FROM events
    """,
    "orders": """
        SELECT id, user_id, event_id, total_amount, status, created_at
        FROM orders
    """,
    "order_items": """
        SELECT id, order_id, ticket_type_id, quantity
        FROM order_items
    """,
    "favorites": """
        SELECT id, user_id, event_id, created_at
        FROM favorites
    """,
    "tickets": """
        SELECT id, order_id, user_id, ticket_type_id, event_id, status, checkin_at, created_at
        FROM tickets
    """,
    "event_ratings": """
        SELECT id, event_id, user_id, rating, created_at
        FROM event_ratings
    """,
    "ticket_types": """
        SELECT id, event_id, name, price, available_quantity, total_quantity, start_date, end_date
        FROM ticket_types
    """,
}


def create_db_engine(settings: Settings) -> Engine | None:
    url = settings.build_database_url()
    if not url:
        logger.info("No database URL configured.")
        return None

    connect_args: dict[str, Any] = {}
    if settings.db_ssl or (settings.db_url and "sslmode=" in settings.db_url):
        connect_args["sslmode"] = "require"
        if settings.db_ssl_ca:
            connect_args["sslrootcert"] = settings.db_ssl_ca
        if not settings.db_ssl_reject_unauthorized:
            connect_args["sslmode"] = "require"

    try:
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # noqa: BLE001 — fallback is intentional
        logger.warning("Could not connect to Postgres: %s", exc)
        return None


def load_corevent_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, query in TABLE_QUERIES.items():
        try:
            frames[name] = pd.read_sql(text(query), engine)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load table %s: %s", name, exc)
            frames[name] = pd.DataFrame()
    return frames


def _has_usable_engagement(tables: dict[str, pd.DataFrame]) -> bool:
    events = tables.get("events", pd.DataFrame())
    orders = tables.get("orders", pd.DataFrame())
    favorites = tables.get("favorites", pd.DataFrame())
    if events.empty:
        return False
    return not orders.empty or not favorites.empty


def load_data(settings: Settings | None = None) -> tuple[dict[str, pd.DataFrame], str]:
    """Load Corevent tables from Postgres, or synthetic data as fallback.

    Returns:
        (tables, source) where source is "postgres" or "synthetic".
    """
    settings = settings or get_settings()
    engine = create_db_engine(settings)
    if engine is not None:
        tables = load_corevent_tables(engine)
        engine.dispose()
        if _has_usable_engagement(tables):
            logger.info(
                "Loaded Postgres data: events=%s orders=%s favorites=%s",
                len(tables["events"]),
                len(tables["orders"]),
                len(tables["favorites"]),
            )
            return tables, "postgres"
        logger.info("Postgres reachable but engagement tables empty; using synthetic data.")

    tables = generate_synthetic_dataset(
        n_users=settings.synthetic_n_users,
        n_events=settings.synthetic_n_events,
        seed=settings.synthetic_seed,
    )
    return tables, "synthetic"
