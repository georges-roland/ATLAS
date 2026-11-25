import pytest

from atlas.config import XAUUSD
from atlas.risk import lots_for_risk, risk_amount, risk_per_lot


def test_risk_per_lot():
    # 10 $ de distance x 100 oz = 1000 $ par lot
    assert risk_per_lot(2000.0, 1990.0, XAUUSD) == 1000.0


def test_lots_for_risk_basic():
    # 10 000 $ x 2 % = 200 $ ; 1000 $/lot -> 0.2 lot
    lots = lots_for_risk(10_000, 2.0, 2000.0, 1990.0, XAUUSD)
    assert lots == pytest.approx(0.2)


def test_lots_rounded_down_never_up():
    # 10 000 x 1 % = 100 $ ; distance 4.3 -> 430 $/lot -> 0.2325 -> 0.23
    lots = lots_for_risk(10_000, 1.0, 2000.0, 1995.7, XAUUSD)
    assert lots == pytest.approx(0.23)
    assert risk_amount(lots, 2000.0, 1995.7, XAUUSD) <= 100.0


def test_below_min_lot_returns_zero():
    # petit compte : 100 $ x 1 % = 1 $ ; 1000 $/lot -> 0.001 lot < 0.01 -> refus
    assert lots_for_risk(100, 1.0, 2000.0, 1990.0, XAUUSD) == 0.0


def test_works_for_sell_side():
    buy = lots_for_risk(10_000, 1.0, 2000.0, 1990.0, XAUUSD)
    sell = lots_for_risk(10_000, 1.0, 2000.0, 2010.0, XAUUSD)
    assert buy == sell == pytest.approx(0.1)


def test_actual_risk_never_exceeds_budget_random_cases():
    import random

    rnd = random.Random(42)
    for _ in range(500):
        equity = rnd.uniform(500, 100_000)
        pct = rnd.uniform(0.1, 3.0)
        entry = rnd.uniform(1500, 3000)
        stop = entry + rnd.choice([-1, 1]) * rnd.uniform(1, 40)
        lots = lots_for_risk(equity, pct, entry, stop, XAUUSD)
        assert risk_amount(lots, entry, stop, XAUUSD) <= equity * pct / 100 + 1e-6


@pytest.mark.parametrize(
    "equity,pct,entry,stop",
    [
        (0, 1, 2000, 1990),
        (-5, 1, 2000, 1990),
        (10_000, 0, 2000, 1990),
        (10_000, 101, 2000, 1990),
        (10_000, 1, 0, 1990),
        (10_000, 1, 2000, 2000),
    ],
)
def test_invalid_inputs_raise(equity, pct, entry, stop):
    with pytest.raises(ValueError):
        lots_for_risk(equity, pct, entry, stop, XAUUSD)
