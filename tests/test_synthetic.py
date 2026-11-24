from corevent_analytics.data.generate_synthetic import generate_synthetic_dataset
from corevent_analytics.data.schema import EVENT_CATEGORIES, OrderStatus


def test_synthetic_is_reproducible() -> None:
    a = generate_synthetic_dataset(n_users=80, n_events=30, seed=7)
    b = generate_synthetic_dataset(n_users=80, n_events=30, seed=7)
    assert list(a.keys()) == list(b.keys())
    for key in a:
        assert a[key].equals(b[key])


def test_synthetic_has_corevent_shape() -> None:
    tables = generate_synthetic_dataset(n_users=120, n_events=40, seed=1)
    assert not tables["users"].empty
    assert not tables["events"].empty
    assert not tables["favorites"].empty
    assert not tables["orders"].empty
    assert set(tables["events"]["category"]).issubset(set(EVENT_CATEGORIES))
    assert OrderStatus.PAID.value in set(tables["orders"]["status"])
    assert {"id", "birth_date", "created_at"} <= set(tables["users"].columns)
    assert "email" not in tables["users"].columns
    assert "qr_code_hash" not in tables["tickets"].columns
