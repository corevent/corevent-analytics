from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from corevent_analytics.data.schema import EVENT_CATEGORIES, OrderStatus, TicketStatus


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _user_age_years(birth_date: pd.Series, reference: datetime | None = None) -> pd.Series:
    reference = reference or datetime.now(UTC)
    birth = pd.to_datetime(birth_date, utc=True, errors="coerce")
    age = (reference - birth).dt.days / 365.25
    return age.clip(lower=0, upper=120)


def build_category_transactions(tables: dict[str, pd.DataFrame]) -> list[list[str]]:
    """One transaction per user: unique event categories from paid orders + favorites."""
    events = tables["events"][["id", "category"]].rename(columns={"id": "event_id"})
    category_by_event = events.set_index("event_id")["category"]

    paid_orders = tables["orders"]
    if not paid_orders.empty:
        paid_orders = paid_orders.loc[paid_orders["status"] == OrderStatus.PAID.value]
    favorites = tables["favorites"]

    rows: list[pd.DataFrame] = []
    if not paid_orders.empty:
        rows.append(paid_orders[["user_id", "event_id"]])
    if not favorites.empty:
        rows.append(favorites[["user_id", "event_id"]])
    if not rows:
        return []

    activity = pd.concat(rows, ignore_index=True).drop_duplicates()
    activity = activity.join(category_by_event, on="event_id", how="inner")
    activity = activity.dropna(subset=["category"])

    transactions: list[list[str]] = []
    for _, group in activity.groupby("user_id"):
        cats = sorted(set(group["category"].astype(str)))
        if cats:
            transactions.append(cats)
    return transactions


def build_user_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Engagement feature matrix indexed by user_id (includes category one-hots)."""
    users = tables["users"].copy()
    users["age"] = _user_age_years(users["birth_date"])
    users["account_age_days"] = (
        datetime.now(UTC) - _ensure_datetime(users["created_at"])
    ).dt.days.clip(lower=0)

    orders = tables["orders"].copy()
    paid = (
        orders.loc[orders["status"] == OrderStatus.PAID.value].copy()
        if not orders.empty
        else pd.DataFrame(columns=orders.columns)
    )
    favorites = tables["favorites"]
    tickets = tables["tickets"]
    events = tables["events"][["id", "category"]].rename(columns={"id": "event_id"})

    order_agg = (
        paid.groupby("user_id")
        .agg(
            n_orders=("id", "count"),
            total_spent=("total_amount", "sum"),
            avg_ticket=("total_amount", "mean"),
        )
        .reset_index()
        if not paid.empty
        else pd.DataFrame(columns=["user_id", "n_orders", "total_spent", "avg_ticket"])
    )

    fav_agg = (
        favorites.groupby("user_id").agg(n_favorites=("id", "count")).reset_index()
        if not favorites.empty
        else pd.DataFrame(columns=["user_id", "n_favorites"])
    )

    if not tickets.empty:
        ticket_stats = tickets.groupby("user_id").agg(
            n_tickets=("id", "count"),
            n_checked_in=("status", lambda s: (s == TicketStatus.CHECKED_IN.value).sum()),
        )
        ticket_stats["checkin_rate"] = (
            ticket_stats["n_checked_in"] / ticket_stats["n_tickets"].replace(0, np.nan)
        ).fillna(0.0)
        ticket_agg = ticket_stats.reset_index()
    else:
        ticket_agg = pd.DataFrame(
            columns=["user_id", "n_tickets", "n_checked_in", "checkin_rate"]
        )

    # Category exposure from favorites + paid purchases.
    exposure_parts: list[pd.DataFrame] = []
    if not paid.empty:
        exposure_parts.append(paid[["user_id", "event_id"]])
    if not favorites.empty:
        exposure_parts.append(favorites[["user_id", "event_id"]])

    cat_features = pd.DataFrame({"user_id": users["id"]})
    for cat in EVENT_CATEGORIES:
        cat_features[f"cat_{cat}"] = 0

    if exposure_parts:
        exposure = pd.concat(exposure_parts, ignore_index=True).drop_duplicates()
        exposure = exposure.merge(events, on="event_id", how="inner")
        for cat in EVENT_CATEGORIES:
            mask = exposure["category"] == cat
            counts = exposure.loc[mask].groupby("user_id").size()
            cat_features[f"cat_{cat}"] = cat_features["user_id"].map(counts).fillna(0).astype(int)

    features = users[["id", "age", "account_age_days"]].rename(columns={"id": "user_id"})
    for frame in (order_agg, fav_agg, ticket_agg):
        features = features.merge(frame, on="user_id", how="left")
    features = features.merge(cat_features, on="user_id", how="left")

    numeric_fill = [
        "n_orders",
        "total_spent",
        "avg_ticket",
        "n_favorites",
        "n_tickets",
        "n_checked_in",
        "checkin_rate",
        *[f"cat_{c}" for c in EVENT_CATEGORIES],
    ]
    for col in numeric_fill:
        if col not in features.columns:
            features[col] = 0
        features[col] = features[col].fillna(0)

    features["age"] = features["age"].fillna(features["age"].median())
    features["account_age_days"] = features["account_age_days"].fillna(0)
    features["has_paid_order"] = (features["n_orders"] > 0).astype(int)
    features["has_favorite"] = (features["n_favorites"] > 0).astype(int)
    return features


def build_classification_frame(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Binary target: converted (paid) among users who favorited at least once.

    Users with favorites and at least one paid order → 1
    Users with favorites and no paid order → 0
    Users without favorites are excluded (not in the conversion funnel).
    """
    features = build_user_features(tables)
    funnel = features.loc[features["has_favorite"] == 1].copy()
    if funnel.empty:
        empty_y = pd.Series(dtype=int, name="converted")
        return funnel, empty_y, []

    y = funnel["has_paid_order"].astype(int).rename("converted")
    drop_cols = {
        "user_id",
        "has_paid_order",
        "n_orders",
        "total_spent",
        "avg_ticket",
        "n_tickets",
        "n_checked_in",
        "checkin_rate",
    }
    # Keep favorites + demographics + category interest; drop leakage from purchases.
    feature_cols = [c for c in funnel.columns if c not in drop_cols]
    x = funnel[feature_cols].copy()
    return x, y, feature_cols


def scale_features(
    x: pd.DataFrame,
    scaler: StandardScaler | None = None,
    *,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = scaler or StandardScaler()
    if fit:
        scaled = scaler.fit_transform(x)
    else:
        scaled = scaler.transform(x)
    return scaled, scaler
