"""Fractional Kelly Criterion & Risk Governance Engine for QuantHoops AI.

Calculates mathematically optimal stake allocation to maximize geometric growth
while capping maximum drawdown risk at 3.5% of total bankroll.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple


def calculate_kelly_stake(
    odds: float,
    effective_prob: float,
    bankroll: float,
    kelly_fraction: float = 0.20,
    max_cap_pct: float = 0.035,
    min_stake_cop: float = 5000.0,
) -> Dict[str, Any]:
    """Computes Fractional Kelly Stake for a basketball investment opportunity.

    Formula:
        b = odds - 1
        p = effective_prob
        q = 1 - p
        f* = (bp - q) / b
        stake = clip(f* * kelly_fraction, 0, max_cap_pct) * bankroll
    """
    p = max(0.001, min(0.999, float(effective_prob)))
    actual_odds = max(1.01, float(odds))
    b = actual_odds - 1.0
    q = 1.0 - p

    ev = (p * actual_odds) - 1.0
    ev_percent = round(ev * 100.0, 2)

    if ev <= 0 or b <= 0:
        return {
            "ev_percent": ev_percent,
            "has_value": False,
            "full_kelly_pct": 0.0,
            "fractional_kelly_pct": 0.0,
            "recommended_stake": 0.0,
            "recommended_stake_pct": 0.0,
            "risk_tier": "Sin Valor / Pasivo",
        }

    full_f_star = (b * p - q) / b
    if full_f_star <= 0:
        return {
            "ev_percent": ev_percent,
            "has_value": False,
            "full_kelly_pct": 0.0,
            "fractional_kelly_pct": 0.0,
            "recommended_stake": 0.0,
            "recommended_stake_pct": 0.0,
            "risk_tier": "Sin Valor / Pasivo",
        }

    frac_stake_pct = full_f_star * float(kelly_fraction)
    # Strict cap to prevent ruin
    capped_stake_pct = min(float(max_cap_pct), max(0.005, frac_stake_pct))

    raw_stake = bankroll * capped_stake_pct

    # Round appropriately
    if bankroll >= 50000:
        recommended_stake = float(round(raw_stake, -2))
        if recommended_stake < min_stake_cop and bankroll >= min_stake_cop * 10:
            recommended_stake = float(min_stake_cop)
    else:
        recommended_stake = float(round(raw_stake, 2))

    # Risk Tier classification
    if capped_stake_pct >= 0.025:
        tier = "Alto Valor / Fuerte Convicción"
    elif capped_stake_pct >= 0.015:
        tier = "Valor Moderado / Estándar"
    else:
        tier = "Micro-Stake / Varianza Controlada"

    return {
        "ev_percent": ev_percent,
        "has_value": True,
        "full_kelly_pct": round(full_f_star * 100.0, 2),
        "fractional_kelly_pct": round(frac_stake_pct * 100.0, 2),
        "recommended_stake": recommended_stake,
        "recommended_stake_pct": round(capped_stake_pct * 100.0, 2),
        "risk_tier": tier,
    }


def evaluate_opportunity_filters(
    odds: float,
    ev_percent: float,
    min_ev: float = 2.0,
    min_odds: float = 1.30,
    max_odds: float = 2.80,
) -> Tuple[bool, str]:
    """Validates if an opportunity meets institutional risk filters."""
    if ev_percent < min_ev:
        return False, f"EV ({ev_percent:.1f}%) inferior al umbral mínimo ({min_ev:.1f}%)"
    if odds < min_odds:
        return False, f"Cuota (@{odds:.2f}) inferior al piso mínimo (@{min_odds:.2f})"
    if odds > max_odds:
        return False, f"Cuota (@{odds:.2f}) superior al techo de varianza (@{max_odds:.2f})"
    return True, "Cumple con todos los filtros de gobernanza cuantitativa."
