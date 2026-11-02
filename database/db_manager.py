"""Database Manager for QuantHoops AI (SQLite).

Stores basketball bets, tracks bankroll evolution, and computes key quantitative
performance metrics (ROI, Yield, Win Rate, Drawdown).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

DB_FILE = os.path.join(os.path.dirname(__file__), "quanthoops.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bankroll_config (
                id INTEGER PRIMARY KEY,
                initial_bankroll REAL NOT NULL,
                current_bankroll REAL NOT NULL,
                currency TEXT DEFAULT 'COP',
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                league TEXT NOT NULL,
                match_name TEXT NOT NULL,
                market TEXT NOT NULL,
                selection TEXT NOT NULL,
                odds REAL NOT NULL,
                fair_odds REAL NOT NULL,
                ev_percent REAL NOT NULL,
                oracle_score REAL DEFAULT 0.0,
                stake REAL NOT NULL,
                status TEXT NOT NULL, -- 'Pendiente', 'Ganada', 'Perdida', 'Anulada'
                payout REAL DEFAULT 0.0,
                profit REAL DEFAULT 0.0,
                bookmaker TEXT DEFAULT 'BetPlay',
                is_golden_pick INTEGER DEFAULT 0,
                is_steam INTEGER DEFAULT 0,
                notes TEXT,
                ai_analysis TEXT
            )
        """)
        # Initialize bankroll if empty
        row = conn.execute("SELECT COUNT(*) as cnt FROM bankroll_config").fetchone()
        if row["cnt"] == 0:
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO bankroll_config (id, initial_bankroll, current_bankroll, currency, updated_at) VALUES (1, 1000000.0, 1000000.0, 'COP', ?)",
                (now,),
            )


init_db()


def get_bankroll_config() -> Dict[str, Any]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM bankroll_config WHERE id = 1").fetchone()
    if row:
        return dict(row)
    return {"initial_bankroll": 1000000.0, "current_bankroll": 1000000.0, "currency": "COP"}


def set_initial_bankroll(amount: float, currency: str = "COP") -> None:
    conn = get_connection()
    now = datetime.now().isoformat()
    with conn:
        conn.execute(
            "UPDATE bankroll_config SET initial_bankroll = ?, current_bankroll = ?, currency = ?, updated_at = ? WHERE id = 1",
            (amount, amount, currency, now),
        )


def calculate_current_bankroll() -> float:
    cfg = get_bankroll_config()
    init_bk = cfg["initial_bankroll"]
    conn = get_connection()
    rows = conn.execute("SELECT profit, stake, status FROM bets").fetchall()
    profit_sum = 0.0
    for r in rows:
        if r["status"] == "Ganada":
            profit_sum += r["profit"]
        elif r["status"] == "Perdida":
            profit_sum -= r["stake"]
    return round(init_bk + profit_sum, 2)


def add_bet(
    league: str,
    match_name: str,
    market: str,
    selection: str,
    odds: float,
    fair_odds: float,
    ev_percent: float,
    stake: float,
    oracle_score: float = 0.0,
    bookmaker: str = "BetPlay",
    is_golden_pick: bool = False,
    is_steam: bool = False,
    notes: str = "",
    ai_analysis: str = "",
    status: str = "Pendiente",
) -> int:
    conn = get_connection()
    now = datetime.now().isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO bets (
                timestamp, league, match_name, market, selection,
                odds, fair_odds, ev_percent, oracle_score, stake, status,
                bookmaker, is_golden_pick, is_steam, notes, ai_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now, league, match_name, market, selection,
                odds, fair_odds, ev_percent, oracle_score, stake, status,
                bookmaker, 1 if is_golden_pick else 0, 1 if is_steam else 0,
                notes, ai_analysis
            ),
        )
        return cursor.lastrowid


def update_bet_status(bet_id: int, new_status: str) -> None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    if not row:
        return

    odds = float(row["odds"])
    stake = float(row["stake"])
    if new_status == "Ganada":
        payout = round(stake * odds, 2)
        profit = round(payout - stake, 2)
    elif new_status == "Perdida":
        payout = 0.0
        profit = -stake
    else:  # Anulada / Pendiente
        payout = stake if new_status == "Anulada" else 0.0
        profit = 0.0

    with conn:
        conn.execute(
            "UPDATE bets SET status = ?, payout = ?, profit = ? WHERE id = ?",
            (new_status, payout, profit, bet_id),
        )


def delete_bet(bet_id: int) -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))


def get_all_bets() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM bets ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_performance_stats() -> Dict[str, Any]:
    bets = get_all_bets()
    total_bets = len(bets)
    resolved_bets = [b for b in bets if b["status"] in ("Ganada", "Perdida")]
    settled_count = len(resolved_bets)

    if settled_count == 0:
        return {
            "total_bets": total_bets,
            "settled_count": 0,
            "won_count": 0,
            "lost_count": 0,
            "win_rate": 0.0,
            "total_staked": sum(b["stake"] for b in bets),
            "net_profit": 0.0,
            "roi_percent": 0.0,
            "yield_percent": 0.0,
        }

    won_count = sum(1 for b in resolved_bets if b["status"] == "Ganada")
    lost_count = sum(1 for b in resolved_bets if b["status"] == "Perdida")
    win_rate = round((won_count / settled_count) * 100.0, 1)

    total_staked_settled = sum(b["stake"] for b in resolved_bets)
    net_profit = sum(b["profit"] for b in resolved_bets)
    yield_pct = round((net_profit / total_staked_settled * 100.0), 2) if total_staked_settled > 0 else 0.0

    cfg = get_bankroll_config()
    init_bk = cfg["initial_bankroll"]
    roi_pct = round((net_profit / init_bk * 100.0), 2) if init_bk > 0 else 0.0

    return {
        "total_bets": total_bets,
        "settled_count": settled_count,
        "won_count": won_count,
        "lost_count": lost_count,
        "win_rate": win_rate,
        "total_staked": round(sum(b["stake"] for b in bets), 2),
        "total_staked_settled": round(total_staked_settled, 2),
        "net_profit": round(net_profit, 2),
        "roi_percent": roi_pct,
        "yield_percent": yield_pct,
    }


def get_bankroll_evolution() -> pd.DataFrame:
    conn = get_connection()
    rows = conn.execute("SELECT id, timestamp, status, stake, profit FROM bets WHERE status IN ('Ganada', 'Perdida') ORDER BY id ASC").fetchall()
    cfg = get_bankroll_config()
    init_bk = cfg["initial_bankroll"]

    history = [{"step": 0, "timestamp": "Inicio", "bankroll": init_bk, "profit": 0.0}]
    curr = init_bk
    for idx, r in enumerate(rows):
        curr += float(r["profit"])
        history.append({
            "step": idx + 1,
            "timestamp": r["timestamp"][:16].replace("T", " "),
            "bankroll": round(curr, 2),
            "profit": float(r["profit"]),
        })
    return pd.DataFrame(history)
