"""Basketball Odds API Client & Smart Money Tracker for QuantHoops AI.

Connects to The Odds API v4 (basketball_nba, basketball_euroleague),
tracks sharp lines (Pinnacle/Bet365), and implements Steam Chasing cache.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests

try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=14400, show_spinner=False)
except Exception:
    def cache_decorator(func):
        return func

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

BASKETBALL_SPORTS: Dict[str, Dict[str, Any]] = {
    "basketball_nba": {
        "key": "basketball_nba",
        "name": "NBA (National Basketball Association)",
        "emoji": "🏀",
        "quarter_minutes": 12,
        "total_minutes": 48,
    },
    "basketball_euroleague": {
        "key": "basketball_euroleague",
        "name": "Turkish Airlines EuroLeague",
        "emoji": "🇪🇺",
        "quarter_minutes": 10,
        "total_minutes": 40,
    },
}

# ==============================================================================
# SMART MONEY & STEAM CHASING IN BASKETBALL
# ==============================================================================

_HOOPS_STEAM_CACHE: Dict[str, Dict[str, Any]] = {}


def extract_sharp_hoops_odds(event: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], str]:
    """Extracts Moneyline odds from Pinnacle, Bet365 or consensus."""
    h_team = event.get("home_team", "")
    a_team = event.get("away_team", "")
    bookmakers = event.get("bookmakers", [])

    pinnacle = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
    bet365 = next((b for b in bookmakers if b.get("key") in ("bet365", "betonlineag")), None)
    target_bm = pinnacle or bet365 or (bookmakers[0] if bookmakers else None)

    if not target_bm:
        return None, "None"

    bm_name = target_bm.get("title", target_bm.get("key", "Sharp"))
    odds_map: Dict[str, float] = {}

    for m in target_bm.get("markets", []):
        if m.get("key") == "h2h":
            for o in m.get("outcomes", []):
                if o.get("name") == h_team:
                    odds_map["1"] = float(o.get("price", 0.0))
                elif o.get("name") == a_team:
                    odds_map["2"] = float(o.get("price", 0.0))
            break

    if "1" in odds_map and "2" in odds_map:
        return odds_map, bm_name
    return None, bm_name


def update_hoops_steam_cache(event: Dict[str, Any]) -> Dict[str, Any]:
    """Caches opening lines and tracks odds drops in basketball."""
    h = event.get("home_team", "")
    a = event.get("away_team", "")
    ev_id = str(event.get("id") or f"{h} vs {a}")
    now = datetime.now()
    curr_odds, bm_name = extract_sharp_hoops_odds(event)

    sim_open = event.get("opening_odds")
    sim_time_ago = float(event.get("opening_hours_ago", 2.5))

    if ev_id not in _HOOPS_STEAM_CACHE:
        if sim_open:
            _HOOPS_STEAM_CACHE[ev_id] = {
                "first_seen": now - timedelta(hours=sim_time_ago),
                "open_odds": dict(sim_open),
                "last_seen": now,
                "current_odds": curr_odds if curr_odds else dict(sim_open),
                "bookmaker": bm_name,
                "home": h,
                "away": a,
            }
        else:
            if curr_odds:
                _HOOPS_STEAM_CACHE[ev_id] = {
                    "first_seen": now,
                    "open_odds": dict(curr_odds),
                    "last_seen": now,
                    "current_odds": dict(curr_odds),
                    "bookmaker": bm_name,
                    "home": h,
                    "away": a,
                }
    else:
        cached = _HOOPS_STEAM_CACHE[ev_id]
        cached["last_seen"] = now
        if curr_odds:
            cached["current_odds"] = dict(curr_odds)
            cached["bookmaker"] = bm_name

    return _HOOPS_STEAM_CACHE.get(ev_id, {})


def detect_hoops_steam_moves(
    event_or_odds: Any,
    window_hours: float = 4.0,
    threshold_pct: float = 10.0,
) -> Dict[str, Any]:
    """Detects line movements (>10% drop in last 4 hours) in basketball.
    Signifies major market movements: late scratch of star player, lineup changes,
    or syndicate sharp money in Pinnacle.
    """
    if isinstance(event_or_odds, dict) and ("home_team" in event_or_odds or "bookmakers" in event_or_odds):
        cached = update_hoops_steam_cache(event_or_odds)
    elif isinstance(event_or_odds, str) and event_or_odds in _HOOPS_STEAM_CACHE:
        cached = _HOOPS_STEAM_CACHE[event_or_odds]
    elif isinstance(event_or_odds, dict) and "open_odds" in event_or_odds:
        cached = event_or_odds
    else:
        cached = {}

    if not cached or "open_odds" not in cached or "current_odds" not in cached:
        return {
            "is_steam": False,
            "steam_detected": False,
            "drop_pct": 0.0,
            "side": None,
            "team": "",
            "open_odds": 0.0,
            "current_odds": 0.0,
            "hours_elapsed": 0.0,
            "badge": "⚪ LÍNEA ESTABLE",
            "description": "Sin datos de apertura suficientes.",
            "bonus_points": 0.0,
        }

    first_seen = cached["first_seen"]
    last_seen = cached["last_seen"]
    hours_elapsed = max(0.1, (last_seen - first_seen).total_seconds() / 3600.0)

    open_o = cached["open_odds"]
    curr_o = cached["current_odds"]
    h_team = cached.get("home", "Local")
    a_team = cached.get("away", "Visitante")

    max_drop = 0.0
    favored_side = None
    favored_team = ""
    open_val = 0.0
    curr_val = 0.0

    for side, t_name in [("1", h_team), ("2", a_team)]:
        if side in open_o and side in curr_o:
            o_init = float(open_o[side])
            o_now = float(curr_o[side])
            if o_init > 1.0 and o_now < o_init:
                drop = ((o_init - o_now) / o_init) * 100.0
                if drop > max_drop:
                    max_drop = drop
                    favored_side = side
                    favored_team = t_name
                    open_val = o_init
                    curr_val = o_now

    is_steam = (max_drop >= threshold_pct) and (hours_elapsed <= window_hours)

    if is_steam:
        badge = "🚨 ALERTA SMART MONEY (Baja de Estrella / Dinero Institucional)"
        desc = (
            f"Caída de cuota institucional de {max_drop:.1f}% en {favored_team} "
            f"(@{open_val:.2f} ➔ @{curr_val:.2f}) en las últimas {hours_elapsed:.1f}h. "
            f"Posible descanso de figura clave del rival o fuerte dinero sharp en Pinnacle."
        )
        bonus = min(100.0, max(20.0, max_drop * 5.0))
    else:
        badge = "⚪ LÍNEA ESTABLE"
        desc = "Fluctuación regular del mercado sin caídas abruptas de cuota."
        bonus = 0.0

    return {
        "is_steam": is_steam,
        "steam_detected": is_steam,
        "drop_pct": round(max_drop, 1),
        "side": favored_side,
        "team": favored_team,
        "open_odds": open_val,
        "current_odds": curr_val,
        "hours_elapsed": round(hours_elapsed, 1),
        "bookmaker": cached.get("bookmaker", "Pinnacle"),
        "badge": badge,
        "description": desc,
        "bonus_points": round(bonus, 1),
    }


# ==============================================================================
# REALISTIC SIMULATION DATASETS
# ==============================================================================

def get_basketball_simulation_matches(sport_key: str) -> List[Dict[str, Any]]:
    """Generates realistic simulation games for NBA and EuroLeague."""
    now = datetime.now(timezone.utc)
    t1 = (now + timedelta(days=1, hours=1)).isoformat()
    t2 = (now + timedelta(days=1, hours=3, minutes=30)).isoformat()
    t3 = (now + timedelta(days=1, hours=6)).isoformat()
    t4 = (now + timedelta(days=2, hours=1)).isoformat()
    t5 = (now + timedelta(days=2, hours=4)).isoformat()

    if sport_key == "basketball_euroleague":
        fixtures = [
            ("Real Madrid", "FC Barcelona", 1.45, 2.80, 1.52, 2.55, -5.5, 163.5, t1, True),
            ("Panathinaikos", "Olympiacos", 1.58, 2.45, 1.68, 2.25, -3.5, 158.5, t2, False),
            ("AS Monaco", "Fenerbahce", 1.65, 2.30, 1.75, 2.15, -2.5, 165.5, t3, False),
            ("Real Madrid", "Olympiacos", 1.38, 3.10, 1.45, 2.80, -6.5, 160.5, t4, False),
        ]
        league_title = "EuroLeague"
    else:
        # NBA
        fixtures = [
            ("Boston Celtics", "Los Angeles Lakers", 1.35, 3.30, 1.42, 2.95, -7.5, 226.5, t1, True),
            ("Denver Nuggets", "Golden State Warriors", 1.48, 2.75, 1.58, 2.45, -5.5, 224.5, t2, True),
            ("Dallas Mavericks", "Milwaukee Bucks", 1.82, 2.05, 1.95, 1.88, -1.5, 231.5, t3, False),
            ("Oklahoma City Thunder", "Minnesota Timberwolves", 1.68, 2.25, 1.78, 2.08, -2.5, 218.5, t4, False),
            ("Miami Heat", "New York Knicks", 2.10, 1.78, 2.25, 1.68, +2.5, 212.5, t5, False),
        ]
        league_title = "NBA"

    events: List[Dict[str, Any]] = []
    for idx, (h, a, pin_1, pin_2, bp_1, bp_2, spread, tot, t_comm, has_steam) in enumerate(fixtures):
        ev: Dict[str, Any] = {
            "id": f"sim_hoops_{sport_key}_{idx+1}",
            "sport_key": sport_key,
            "sport_title": league_title,
            "commence_time": t_comm,
            "home_team": h,
            "away_team": a,
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle (Oráculo)",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [{"name": h, "price": pin_1}, {"name": a, "price": pin_2}],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [{"name": h, "price": 1.91, "point": spread}, {"name": a, "price": 1.91, "point": -spread}],
                        },
                        {
                            "key": "totals",
                            "outcomes": [{"name": "Over", "price": 1.90, "point": tot}, {"name": "Under", "price": 1.90, "point": tot}],
                        },
                    ],
                },
                {
                    "key": "betplay",
                    "title": "BetPlay",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [{"name": h, "price": bp_1}, {"name": a, "price": bp_2}],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [{"name": h, "price": 1.88, "point": spread}, {"name": a, "price": 1.88, "point": -spread}],
                        },
                        {
                            "key": "totals",
                            "outcomes": [{"name": "Over", "price": 1.87, "point": tot}, {"name": "Under", "price": 1.87, "point": tot}],
                        },
                    ],
                },
            ],
        }

        # Seed Steam Moves on selected games
        if has_steam:
            ev["opening_odds"] = {"1": round(pin_1 * 1.15, 2), "2": round(pin_2 * 0.90, 2)}
            ev["opening_hours_ago"] = 2.5

        update_hoops_steam_cache(ev)
        ev["steam_move"] = detect_hoops_steam_moves(ev)
        events.append(ev)

    return events


@cache_decorator
def fetch_basketball_odds(
    sport_key: str = "basketball_nba",
    api_key: Optional[str] = None,
    regions: str = "eu,us",
    markets: str = "h2h,spreads,totals",
    is_demo: bool = True,
    force_nonce: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """Fetches real-time basketball odds from The Odds API or provides realistic simulation data."""
    metadata = {
        "requests_remaining": "N/A",
        "requests_used": "N/A",
        "source": "demo" if is_demo else "api",
        "sport_key": sport_key,
    }

    if is_demo or not api_key or not api_key.strip():
        metadata["requests_remaining"] = "500 (Modo Demo Activo)"
        metadata["requests_used"] = "0"
        return get_basketball_simulation_matches(sport_key), metadata, None

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key.strip(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        try:
            resp = requests.get(url, params=params, timeout=12)
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(url, params=params, timeout=12, verify=False)

        if "x-requests-remaining" in resp.headers:
            metadata["requests_remaining"] = resp.headers.get("x-requests-remaining")
        if "x-requests-used" in resp.headers:
            metadata["requests_used"] = resp.headers.get("x-requests-used")

        if resp.status_code == 200:
            data = resp.json()
            if not data:
                return (
                    get_basketball_simulation_matches(sport_key),
                    metadata,
                    f"No hay partidos activos en The Odds API para {sport_key}. Cargando simulación.",
                )
            for ev in data:
                update_hoops_steam_cache(ev)
                ev["steam_move"] = detect_hoops_steam_moves(ev)
            return data, metadata, None

        return (
            get_basketball_simulation_matches(sport_key),
            metadata,
            f"Error {resp.status_code} desde The Odds API. Mostrando simulación.",
        )
    except Exception as e:
        return (
            get_basketball_simulation_matches(sport_key),
            metadata,
            f"Error de conexión ({type(e).__name__}). Mostrando simulación.",
        )


def format_short_time(iso_str: str) -> str:
    """Formats ISO datetime to readable format."""
    if not iso_str:
        return "Próximamente"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        local_dt = dt - timedelta(hours=5) # UTC-5 Colombia
        return local_dt.strftime("%d/%m %I:%M %p")
    except Exception:
        return iso_str[:16].replace("T", " ")
