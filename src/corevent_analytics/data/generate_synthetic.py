from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from corevent_analytics.data.schema import (
    CATEGORY_AFFINITIES,
    EVENT_CATEGORIES,
    EVENT_LOCATION_TYPES,
    EVENT_STATUSES,
    OrderStatus,
    TicketStatus,
)


def _uuid_series(n: int, rng: np.random.Generator) -> list[str]:
    """Deterministic UUID-like strings from a seeded RNG."""
    return [
        f"{rng.integers(0, 2**32):08x}-{rng.integers(0, 2**16):04x}-"
        f"4{rng.integers(0, 2**12):03x}-a{rng.integers(0, 2**12):03x}-"
        f"{rng.integers(0, 2**48):012x}"
        for _ in range(n)
    ]


def generate_synthetic_dataset(
    n_users: int = 3000,
    n_events: int = 200,
    seed: int = 42,
    reference_time: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a Corevent-shaped dataset with category co-occurrence for Apriori."""
    rng = np.random.default_rng(seed)
    # Fixed default keeps datasets reproducible across calls with the same seed.
    now = reference_time or datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

    user_ids = _uuid_series(n_users, rng)
    ages = rng.integers(16, 65, size=n_users)
    birth_dates = [
        (now - timedelta(days=int(age * 365.25) + int(rng.integers(0, 365)))).date()
        for age in ages
    ]
    users = pd.DataFrame(
        {
            "id": user_ids,
            "birth_date": birth_dates,
            "created_at": [
                now - timedelta(days=int(rng.integers(30, 900))) for _ in range(n_users)
            ],
        }
    )

    event_ids = _uuid_series(n_events, rng)
    # Over-sample affinity hubs so co-occurrence is natural.
    hub_categories = list(CATEGORY_AFFINITIES.keys())
    category_weights = np.array(
        [3.0 if c in hub_categories else 1.0 for c in EVENT_CATEGORIES],
        dtype=float,
    )
    category_weights /= category_weights.sum()
    categories = rng.choice(EVENT_CATEGORIES, size=n_events, p=category_weights)
    events = pd.DataFrame(
        {
            "id": event_ids,
            "category": categories,
            "location_type": rng.choice(EVENT_LOCATION_TYPES, size=n_events),
            "city_id": rng.integers(1, 500, size=n_events),
            "is_adult_only": rng.random(n_events) < 0.15,
            "status": rng.choice(
                [
                    s
                    for s in EVENT_STATUSES
                    if s not in {"draft"}
                ],
                size=n_events,
            ),
            "start_date": [
                now + timedelta(days=int(rng.integers(-60, 120))) for _ in range(n_events)
            ],
            "end_date": [
                now + timedelta(days=int(rng.integers(-59, 121))) for _ in range(n_events)
            ],
            "created_at": [
                now - timedelta(days=int(rng.integers(10, 400))) for _ in range(n_events)
            ],
        }
    )

    events_by_category: dict[str, list[str]] = {
        cat: events.loc[events["category"] == cat, "id"].tolist() for cat in EVENT_CATEGORIES
    }

    ticket_type_rows: list[dict[str, object]] = []
    for event_id in event_ids:
        n_types = int(rng.integers(1, 4))
        for i in range(n_types):
            total_qty = int(rng.integers(50, 500))
            ticket_type_rows.append(
                {
                    "id": _uuid_series(1, rng)[0],
                    "event_id": event_id,
                    "name": f"Tipo {i + 1}",
                    "price": float(rng.choice([20, 40, 80, 120, 200, 350])),
                    "available_quantity": int(rng.integers(0, total_qty)),
                    "total_quantity": total_qty,
                    "start_date": now - timedelta(days=30),
                    "end_date": now + timedelta(days=60),
                }
            )
    ticket_types = pd.DataFrame(ticket_type_rows)
    ticket_types_by_event = {
        eid: group["id"].tolist()
        for eid, group in ticket_types.groupby("event_id")
    }
    price_by_ticket_type = ticket_types.set_index("id")["price"].to_dict()

    favorite_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    order_item_rows: list[dict[str, object]] = []
    ticket_rows: list[dict[str, object]] = []
    rating_rows: list[dict[str, object]] = []

    for user_id in user_ids:
        # Pick a primary interest hub for this user.
        primary = hub_categories[int(rng.integers(0, len(hub_categories)))]
        related = list(CATEGORY_AFFINITIES.get(primary, ()))
        interest_cats = [primary, *related]
        if rng.random() < 0.35:
            interest_cats.append(str(rng.choice(EVENT_CATEGORIES)))

        candidate_events: list[str] = []
        for cat in interest_cats:
            pool = events_by_category.get(cat, [])
            if pool:
                take = min(len(pool), int(rng.integers(1, 4)))
                candidate_events.extend(rng.choice(pool, size=take, replace=False).tolist())

        candidate_events = list(dict.fromkeys(candidate_events))
        if not candidate_events:
            continue

        n_fav = min(len(candidate_events), int(rng.integers(1, 5)))
        fav_events = rng.choice(candidate_events, size=n_fav, replace=False).tolist()
        for event_id in fav_events:
            favorite_rows.append(
                {
                    "id": _uuid_series(1, rng)[0],
                    "user_id": user_id,
                    "event_id": event_id,
                    "created_at": now - timedelta(days=int(rng.integers(1, 180))),
                }
            )

        # ~55% of users with favorites convert to at least one paid order.
        converts = rng.random() < 0.55
        purchase_pool = fav_events if converts else []
        if converts and rng.random() < 0.4 and len(candidate_events) > len(fav_events):
            extra = [e for e in candidate_events if e not in fav_events]
            if extra:
                purchase_pool = fav_events + [str(rng.choice(extra))]

        for event_id in purchase_pool:
            order_id = _uuid_series(1, rng)[0]
            tt_ids = ticket_types_by_event.get(event_id, [])
            if not tt_ids:
                continue
            ticket_type_id = str(rng.choice(tt_ids))
            qty = int(rng.integers(1, 4))
            unit_price = float(price_by_ticket_type[ticket_type_id])
            status = OrderStatus.PAID.value if rng.random() < 0.85 else str(
                rng.choice(
                    [
                        OrderStatus.PENDING.value,
                        OrderStatus.CANCELLED.value,
                        OrderStatus.FAILED.value,
                    ]
                )
            )
            order_rows.append(
                {
                    "id": order_id,
                    "user_id": user_id,
                    "event_id": event_id,
                    "total_amount": unit_price * qty,
                    "status": status,
                    "created_at": now - timedelta(days=int(rng.integers(1, 120))),
                }
            )
            order_item_rows.append(
                {
                    "id": _uuid_series(1, rng)[0],
                    "order_id": order_id,
                    "ticket_type_id": ticket_type_id,
                    "quantity": qty,
                }
            )
            if status != OrderStatus.PAID.value:
                continue
            for _ in range(qty):
                checked_in = rng.random() < 0.7
                ticket_rows.append(
                    {
                        "id": _uuid_series(1, rng)[0],
                        "order_id": order_id,
                        "user_id": user_id,
                        "ticket_type_id": ticket_type_id,
                        "event_id": event_id,
                        "status": (
                            TicketStatus.CHECKED_IN.value
                            if checked_in
                            else TicketStatus.PENDING.value
                        ),
                        "checkin_at": now - timedelta(hours=int(rng.integers(1, 72)))
                        if checked_in
                        else None,
                        "created_at": now - timedelta(days=int(rng.integers(1, 120))),
                    }
                )
            if rng.random() < 0.4:
                rating_rows.append(
                    {
                        "id": _uuid_series(1, rng)[0],
                        "event_id": event_id,
                        "user_id": user_id,
                        "rating": int(rng.integers(3, 6)),
                        "created_at": now - timedelta(days=int(rng.integers(1, 90))),
                    }
                )

    return {
        "users": users,
        "events": events,
        "orders": pd.DataFrame(order_rows),
        "order_items": pd.DataFrame(order_item_rows),
        "favorites": pd.DataFrame(favorite_rows),
        "tickets": pd.DataFrame(ticket_rows),
        "event_ratings": pd.DataFrame(rating_rows),
        "ticket_types": ticket_types,
    }
