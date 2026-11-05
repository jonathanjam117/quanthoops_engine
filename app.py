"""QuantHoops AI — Motor Cuantitativo Institucional de Baloncesto (NBA & EuroLeague).

Arquitectura Gaussiana de 3 Motores:
- Motor 1: Rastreador de Smart Money & Caídas de Línea (Steam Chasing)
- Motor 2: Simulador Monte Carlo Bivariado Gaussiano (10,000 Iteraciones)
- Motor 3: Score de Oráculo (0-100) con Gobernanza Estricta (>65 / +EV >= 2.0%)
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Reads secrets from st.secrets (Streamlit Cloud) with fallback to os.getenv."""
    try:
        if hasattr(st, "secrets"):
            if key in st.secrets:
                val = st.secrets[key]
                if val is not None and str(val).strip():
                    return str(val).strip()
            for sec in st.secrets:
                sec_obj = st.secrets[sec]
                if isinstance(sec_obj, dict) and key in sec_obj:
                    val = sec_obj[key]
                    if val is not None and str(val).strip():
                        return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, default)


# Module imports
from database.db_manager import (
    add_bet,
    calculate_current_bankroll,
    delete_bet,
    get_all_bets,
    get_bankroll_config,
    get_bankroll_evolution,
    get_performance_stats,
    set_initial_bankroll,
    update_bet_status,
)
from etl.basketball_odds_client import (
    BASKETBALL_SPORTS,
    detect_hoops_steam_moves,
    extract_sharp_hoops_odds,
    fetch_basketball_odds,
    format_short_time,
)
from models.basketball_engine import (
    LEAGUE_BENCHMARKS,
    analyze_basketball_match_opportunities,
    calculate_expected_points,
    generate_basketball_sgp_cards,
    get_team_profile,
    run_basketball_monte_carlo,
)
from models.kelly_risk import calculate_kelly_stake, evaluate_opportunity_filters

# Streamlit Page Config
st.set_page_config(
    page_title="QuantHoops AI | Basketball Oracle Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def format_money(amount: float, currency: str = "COP") -> str:
    """Formats numeric values into localized currency."""
    if currency == "COP":
        return f"${amount:,.0f} COP".replace(",", ".")
    elif currency == "USD":
        return f"${amount:,.2f} USD"
    return f"€{amount:,.2f} EUR"


def clean_html(raw_html: str) -> str:
    """Eliminates indentation artifacts that break Streamlit Markdown."""
    return "\n".join([line.lstrip() for line in raw_html.splitlines() if line.strip()])


# Global Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b1120; }
    div[data-testid="stSidebar"] {
        background-color: #0d1527;
        border-right: 1px solid #1e293b;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    button[data-baseweb="tab"] {
        font-weight: 700;
        font-size: 0.95rem;
        color: #94a3b8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS & GOVERNANCE
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 6px 0 14px 0;">
        <div style="font-size: 2.3rem;">🏀</div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff;">QuantHoops AI</div>
        <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 700;">MOTOR ORÁCULO GAUSSIANO</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏦 Control de Banca")
    bk_cfg = get_bankroll_config()
    current_bk_calc = calculate_current_bankroll()

    curr_col1, curr_col2 = st.columns(2)
    with curr_col1:
        selected_curr = st.selectbox(
            "Moneda",
            options=["COP", "USD", "EUR"],
            index=["COP", "USD", "EUR"].index(bk_cfg.get("currency", "COP")),
        )
    with curr_col2:
        st.metric("Banca Activa", format_money(current_bk_calc, selected_curr))

    kelly_mult = st.slider(
        "Fracción de Kelly (k)",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="0.20 representa Fifth-Kelly (un quinto de Kelly), el estándar para minimizar la racha de varianza.",
    )

    min_ev_thresh = st.slider(
        "Filtro Mínimo +EV (%)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Solo destaca selecciones con ventaja matemática positiva superior a este umbral.",
    )

    odds_range = st.slider(
        "Rango Eficiente de Cuotas",
        min_value=1.20,
        max_value=3.50,
        value=(1.30, 2.80),
        step=0.05,
        help="Rango calibrado institucional para alto volumen y convexidad positiva.",
    )

    st.caption(
        f"🔒 **Regla de Oro**: Tope máximo fijado en **3.5% del Bankroll** "
        f"({format_money(current_bk_calc * 0.035, selected_curr)})."
    )

    st.markdown("---")
    st.markdown("### 🔌 Conexión The Odds API")
    env_odds_key = _get_secret("THE_ODDS_API_KEY", "")
    has_key = bool(env_odds_key)

    demo_mode = st.toggle("🧪 Modo Simulación Realista", value=not has_key)
    api_key_input = st.text_input("The Odds API Key", value=env_odds_key, type="password")

    st.markdown("---")
    if st.sidebar.button("🔄 Actualizar Cuotas (API)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("QuantHoops AI v1.0 • Motor Bivariado Gaussiano • Steam Chasing")

# ==============================================================================
# HEADER & TOURNAMENT SELECTOR
# ==============================================================================
col_top1, col_top2 = st.columns([3, 1])
with col_top1:
    selected_league_key = st.selectbox(
        "🏆 Liga / Competición Activa:",
        options=list(BASKETBALL_SPORTS.keys()),
        format_func=lambda k: f"{BASKETBALL_SPORTS[k]['emoji']} {BASKETBALL_SPORTS[k]['name']}",
        index=0,
    )
with col_top2:
    st.write("")
    st.write("")
    refresh_btn = st.button("🔄 Actualizar Cuotas", use_container_width=True)
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

league_meta = BASKETBALL_SPORTS[selected_league_key]

# Fetch Odds
events, meta, err_msg = fetch_basketball_odds(
    sport_key=selected_league_key,
    api_key=api_key_input if not demo_mode else None,
    is_demo=demo_mode,
)

if err_msg:
    st.warning(f"ℹ️ {err_msg}")

# Banner Mode
if meta.get("source") == "demo":
    st.info("🧪 **Modo Simulación Activo:** Mostrando cartelera con cuotas institucionales de Pinnacle y movimientos de Smart Money pre-cargados.")
else:
    st.success(f"🟢 **Conectado a The Odds API en Vivo:** Cuotas reales de casas de referencia global. Restantes: {meta.get('requests_remaining', 'N/A')}")

# ==============================================================================
# TABS PRINCIPALES
# ==============================================================================
tab_scanner, tab_betplay, tab_stats, tab_history = st.tabs([
    "🎯 Escáner Oráculo (+EV)",
    "🏀 BetPlay (Crear Apuesta)",
    "📊 Análisis Gaussiano (Pace & ORtg)",
    "📈 Control de Banca (SQLite)",
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: ESCÁNER ORÁCULO (+EV)
# ------------------------------------------------------------------------------
with tab_scanner:
    st.subheader(f"🎯 Señales Cuantitativas ({league_meta['name']}) — Modo Oráculo Gaussiano")
    st.caption(
        "Filtrado estricto: **Score de Oráculo > 65/100**, **EV ≥ 2.0%**, **Cuotas [1.30, 2.80]** y **Simulación Normal (10,000 iteraciones)**."
    )

    detected_opportunities = []
    fixture_rows = []

    for ev in events:
        h = ev.get("home_team", "")
        a = ev.get("away_team", "")
        t_str = format_short_time(ev.get("commence_time", ""))
        steam_data = ev.get("steam_move", {})

        fixture_rows.append({
            "Partido": f"{h} vs {a}",
            "Competición": league_meta["name"],
            "Horario": t_str,
            "Smart Money": steam_data.get("badge", "⚪ ESTABLE"),
        })

        opps = analyze_basketball_match_opportunities(
            event=ev,
            sport_meta=league_meta,
            bankroll=current_bk_calc,
            kelly_k=kelly_mult,
            min_ev=min_ev_thresh,
            min_odds=odds_range[0],
            max_odds=odds_range[1],
        )
        detected_opportunities.extend(opps)

    # Filter with Governance Rule: Score > 65.0, EV >= min_ev, Odds in range, Stake > 0
    min_qual_ev = float(min_ev_thresh)
    min_o, max_o = odds_range
    qualified_signals = [
        opp for opp in detected_opportunities
        if float(opp.get("oracle_score", 0.0)) > 65.0
        and float(opp.get("ev_percent", 0.0)) >= min_qual_ev
        and min_o <= float(opp.get("odds", 0.0)) <= max_o
        and float(opp.get("recommended_stake", 0.0)) > 0
    ]

    qualified_matches = set(opp.get("match", "") for opp in qualified_signals)
    discarded_fixtures = [fix for fix in fixture_rows if fix.get("Partido", "") not in qualified_matches]

    st.markdown("### 🎯 Oportunidades Automáticas con +EV Detectadas Hoy")
    st.caption(
        f"Gobernanza Cuantitativa: **Score > 65/100**, **EV ≥ {min_qual_ev:.1f}%**, **Cuotas [{min_o:.2f}, {max_o:.2f}]**, "
        f"y Criterio de Kelly (k={kelly_mult})."
    )

    if not qualified_signals:
        st.warning(
            f"⚠️ **Sin operaciones 'Luz Verde' hoy:** Ningún partido superó el Score de Oráculo institucional (> 65/100). "
            "El motor Gaussiano protege el 100% de tu capital bloqueando apuestas con ventaja desfavorable o varianza excesiva."
        )
    else:
        golden_count = sum(1 for s in qualified_signals if s.get("is_golden_pick") or s.get("oracle_score", 0) >= 90.0)
        status_msg = f"🟢 **{len(qualified_signals)} Operación(es) Calificada(s) por el Oráculo Gaussiano**"
        if golden_count > 0:
            status_msg += f" — Incluye **{golden_count} 🔥 GOLDEN PICK(S)**"
        st.success(status_msg)

        for s_idx, sig in enumerate(qualified_signals):
            m_title = sig.get("match", "")
            m_market = sig.get("market", "")
            m_sel = sig.get("selection", "")
            m_odds = float(sig.get("odds", 0.0))
            m_ev = float(sig.get("ev_percent", 0.0))
            m_score = float(sig.get("oracle_score", 0.0))
            m_stake = float(sig.get("recommended_stake", 0.0))
            m_time = format_short_time(sig.get("commence_time", ""))
            m_bm = sig.get("bookmaker", "Pinnacle/BetPlay")
            is_gold = bool(sig.get("is_golden_pick", False) or m_score >= 90.0)
            steam_data = sig.get("steam_move", {})
            is_steam = bool(steam_data.get("is_steam", False))
            mc_prob = float(sig.get("monte_carlo_prob", sig.get("effective_prob_pct", 50.0)))

            card_border = "#fbbf24" if is_gold else "#10b981"
            badge_html = (
                '<span style="background: linear-gradient(90deg, #b45309, #d97706); color: #fffbeb; font-weight: 900; font-size: 0.76rem; padding: 3px 9px; border-radius: 6px; border: 1px solid #fef3c7;">🔥 GOLDEN PICK</span>'
                if is_gold else
                '<span style="background-color: #065f46; color: #a7f3d0; font-weight: 800; font-size: 0.76rem; padding: 3px 8px; border-radius: 6px;">🟢 LUZ VERDE (+EV)</span>'
            )

            steam_badge = f'<span style="background: rgba(239, 68, 68, 0.22); border: 1px solid #ef4444; color: #fca5a5; font-weight: 800; font-size: 0.76rem; padding: 3px 8px; border-radius: 6px; margin-left: 6px;">🚨 SMART MONEY (-{steam_data.get("drop_pct", 0)}%)</span>' if is_steam else ''

            html_card = f"""
            <div style="background: linear-gradient(135deg, #0f192d 0%, #172554 100%); border: 1.5px solid {card_border}; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.35);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        {badge_html}
                        <span style="background: rgba(168, 85, 247, 0.20); border: 1px solid #c084fc; color: #e9d5ff; font-weight: 800; font-size: 0.76rem; padding: 3px 8px; border-radius: 6px; margin-left: 6px;">🔮 Score: {m_score:.1f}/100</span>
                        <span style="background: rgba(56, 189, 248, 0.18); border: 1px solid #38bdf8; color: #bae6fd; font-weight: 800; font-size: 0.76rem; padding: 3px 8px; border-radius: 6px; margin-left: 6px;">🎲 Monte Carlo (10k): {mc_prob:.1f}%</span>
                        {steam_badge}
                        <span style="color: #94a3b8; font-size: 0.82rem; margin-left: 8px;">🏀 {league_meta['name']} | 📅 {m_time}</span>
                        <div style="font-size: 1.22rem; font-weight: 800; color: #ffffff; margin-top: 4px;">
                            🏀 {m_title}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.35rem; font-weight: 800; color: #fbbf24;">@{m_odds:.2f}</span>
                        <span style="color: #64748b; font-size: 0.85rem; margin-left: 4px;">({m_bm})</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); flex-wrap: wrap; gap: 8px;">
                    <div style="font-size: 0.95rem; color: #f1f5f9;">
                        👉 <b>Selección:</b> <span style="color: #38bdf8; font-weight: 700;">{m_market} → {m_sel}</span>
                    </div>
                    <div style="display: flex; gap: 14px; align-items: center;">
                        <span style="background: rgba(16 put_0, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; font-weight: 800; font-size: 0.90rem; padding: 3px 10px; border-radius: 6px;">
                            📈 EV: +{m_ev:.1f}%
                        </span>
                        <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; font-weight: 800; font-size: 0.90rem; padding: 3px 10px; border-radius: 6px;">
                            💰 Stake Kelly: {format_money(m_stake, selected_curr)}
                        </span>
                    </div>
                </div>
            </div>
            """
            st.markdown(clean_html(html_card), unsafe_allow_html=True)

            if st.button(f"💾 Guardar Señal #{s_idx+1} en SQLite", key=f"btn_save_hoops_{s_idx}"):
                bid = add_bet(
                    league=league_meta["name"],
                    match_name=m_title,
                    market=m_market,
                    selection=m_sel,
                    odds=m_odds,
                    fair_odds=round(1.0 / (mc_prob / 100.0), 2),
                    ev_percent=m_ev,
                    stake=m_stake,
                    oracle_score=m_score,
                    bookmaker=m_bm,
                    is_golden_pick=is_gold,
                    is_steam=is_steam,
                )
                st.toast(f"✅ ¡Operación #{bid} registrada en SQLite!", icon="💾")

    # Discarded Fixtures
    st.markdown("---")
    with st.expander(f"📁 Partidos Descartados por Filtro de Gobernanza ({len(discarded_fixtures)} eventos)", expanded=False):
        st.caption("Partidos donde el Score de Oráculo fue ≤ 65/100 o el EV fue inferior al umbral. El sistema bloquea el stake ($0) para proteger el capital.")
        if discarded_fixtures:
            st.dataframe(pd.DataFrame(discarded_fixtures), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# PESTAÑA 2: BETPLAY / CREAR APUESTA
# ------------------------------------------------------------------------------
with tab_betplay:
    st.subheader("🏀 BetPlay — Módulo 'Crear Apuesta' (Same-Game Parlays)")
    st.caption("Estrategias combinadas intra-partido resueltas mediante Simulación Monte Carlo Normal (10,000 iteraciones).")

    if not events:
        st.warning("No hay partidos disponibles en este momento.")
    else:
        m_opts = [f"{ev['home_team']} vs {ev['away_team']}" for ev in events]
        sel_match_idx = st.selectbox("Selecciona el Partido para 'Crear Apuesta':", range(len(m_opts)), format_func=lambda i: m_opts[i])
        cur_ev = events[sel_match_idx]

        h = cur_ev["home_team"]
        a = cur_ev["away_team"]
        mu_h, mu_a, pace = calculate_expected_points(h, a, selected_league_key)
        mc_sim = run_basketball_monte_carlo(mu_h, mu_a, league_key=selected_league_key, iterations=10000)
        steam_data = cur_ev.get("steam_move", {})

        # Top Match Banner
        st.markdown(clean_html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 1.5px solid #334155; border-radius: 12px; padding: 16px 20px; margin: 12px 0 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff;">
                        🏀 {h} <span style="color: #64748b; font-weight: 400;">vs</span> {a}
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                        📅 <b>Horario:</b> {format_short_time(cur_ev.get('commence_time', ''))} | ⚡ <b>Pace Estimado:</b> ~{pace} posesiones
                    </div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span style="background-color: #1e3a5f; color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                        Puntuación Esperada: {h} {mu_h:.1f} - {mu_a:.1f} {a}
                    </div>
                    <span style="background-color: #064e3b; color: #34d399; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                        🎯 Total Esperado: ~{mc_sim['avg_total']:.1f} pts
                    </span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # 3 SGP Cards
        sgp_cards = generate_basketball_sgp_cards(
            home_team=h,
            away_team=a,
            mu_h=mu_h,
            mu_a=mu_a,
            mc_sim=mc_sim,
            bankroll=current_bk_calc,
            kelly_multiplier=kelly_mult,
            steam_data=steam_data,
        )

        card_cols = st.columns(3)
        for c_idx, card in enumerate(sgp_cards):
            with card_cols[c_idx]:
                with st.container(border=True):
                    st.markdown(f"### {card['badge']}")
                    st.caption(card["description"])

                    c_score = float(card.get("oracle_score", 0.0))
                    c_gold = bool(card.get("is_golden_pick", False))
                    c_steam = card.get("steam_move", {})

                    if c_gold:
                        st.info("🔥 **GOLDEN PICK (Score ≥ 90/100)**")
                    if c_steam and c_steam.get("is_steam"):
                        st.error(f"🚨 **SMART MONEY**: Caída de {c_steam.get('drop_pct')}%")

                    st.markdown("---")
                    for leg in card["legs"]:
                        st.markdown(f"• **{leg['selection']}** :gray[({leg['market']})] **:orange[@{leg['odds']:.2f}]**")

                    st.markdown("---")
                    sub_c1, sub_c2 = st.columns(2)
                    with sub_c1:
                        st.metric("Cuota Total", f"@{card['odds']:.2f}")
                        st.metric("Probabilidad (10k)", f"{card['prob_pct']}%")
                    with sub_c2:
                        st.metric("Score Oráculo", f"{c_score:.1f}/100")
                        if c_score <= 65.0:
                            st.metric("Stake Kelly", f"🔒 $0 ({selected_curr})")
                            st.caption("🔒 Bloqueado: Score ≤ 65")
                        else:
                            st.metric("Stake Kelly", format_money(card["recommended_stake"], selected_curr))

                    if st.button(f"💾 Guardar {card['badge']}", key=f"btn_save_sgp_hoops_{c_idx}"):
                        summary_txt = " + ".join([l["selection"] for l in card["legs"]])
                        bid = add_bet(
                            league=league_meta["name"],
                            match_name=f"{h} vs {a}",
                            market=card["badge"],
                            selection=summary_txt,
                            odds=card["odds"],
                            fair_odds=round(1.0 / (card["prob_pct"] / 100.0), 2),
                            ev_percent=card["ev_percent"],
                            stake=card["recommended_stake"],
                            oracle_score=c_score,
                            is_golden_pick=c_gold,
                            is_steam=bool(c_steam.get("is_steam")),
                        )
                        st.toast(f"✅ ¡{card['badge']} guardada en SQLite (ID #{bid})!", icon="💾")

# ------------------------------------------------------------------------------
# PESTAÑA 3: ANÁLISIS GAUSSIANO & EFICIENCIA DE POSESIONES
# ------------------------------------------------------------------------------
with tab_stats:
    st.subheader("📊 Análisis Gaussiano & Distribución Bivariada de Puntos")
    st.caption("Demostración cuantitativa: En baloncesto, la suma de posesiones genera una distribución Normal y no de Poisson.")

    if events:
        sel_ev_stats = events[sel_match_idx if 'sel_match_idx' in locals() else 0]
        sh = sel_ev_stats["home_team"]
        sa = sel_ev_stats["away_team"]
        s_mu_h, s_mu_a, s_pace = calculate_expected_points(sh, sa, selected_league_key)
        bench = LEAGUE_BENCHMARKS[selected_league_key]
        sigma = bench["sigma"]

        col_st1, col_st2, col_st3 = st.columns(3)
        with col_st1:
            st.metric(f"Ritmo (Pace) Estimado", f"{s_pace:.1f} pos")
            st.caption(f"Promedio de Liga: {bench['pace']} posesiones")
        with col_st2:
            st.metric(f"Media Proyectada {sh}", f"{s_mu_h:.1f} pts")
            st.caption(f"Desviación Estándar σ = {sigma:.1f} pts")
        with col_st3:
            st.metric(f"Media Proyectada {sa}", f"{s_mu_a:.1f} pts")
            st.caption(f"Desviación Estándar σ = {sigma:.1f} pts")

        # Plot Gaussian Density Curves
        x_pts = np.linspace(min(s_mu_h, s_mu_a) - 35, max(s_mu_h, s_mu_a) + 35, 200)
        pdf_h = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pts - s_mu_h) / sigma) ** 2)
        pdf_a = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pts - s_mu_a) / sigma) ** 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_pts, y=pdf_h, mode="lines", name=f"{sh} (Local)", line=dict(color="#38bdf8", width=3)))
        fig.add_trace(go.Scatter(x=x_pts, y=pdf_a, mode="lines", name=f"{sa} (Visitante)", line=dict(color="#f43f5e", width=3)))
        fig.update_layout(
            title=f"Distribución Normal Teórica de Puntuación: {sh} vs {sa}",
            xaxis_title="Puntos Totales Anotados",
            yaxis_title="Densidad de Probabilidad Gaussiana",
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 4: CONTROL DE BANCA (SQLITE)
# ------------------------------------------------------------------------------
with tab_history:
    st.subheader("📈 Control de Rendimiento & Auditoría de Banca")
    stats = get_performance_stats()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Operaciones", f"{stats['total_bets']}")
    with m_col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with m_col3:
        st.metric("Yield Cuantitativo", f"{stats['yield_percent']:+.1f}%")
    with m_col4:
        st.metric("Beneficio Neto", format_money(stats["net_profit"], selected_curr))

    # Bankroll chart
    evo_df = get_bankroll_evolution()
    if len(evo_df) > 1:
        fig_evo = px.line(
            evo_df,
            x="step",
            y="bankroll",
            title="Evolución Histórica de la Banca",
            template="plotly_dark",
            markers=True,
        )
        fig_evo.update_traces(line_color="#10b981", line_width=3)
        st.plotly_chart(fig_evo, use_container_width=True)

    # Bets table
    all_b = get_all_bets()
    if all_b:
        st.markdown("### 📋 Registro de Inversiones Registradas")
        for b in all_b:
            b_id = b["id"]
            st_col1, st_col2, st_col3, st_col4 = st.columns([3, 1, 1, 1])
            with st_col1:
                st.markdown(f"**#{b_id} {b['match_name']}** — {b['market']}: :blue[{b['selection']}] @:orange[{b['odds']:.2f}]")
                st.caption(f"Fecha: {b['timestamp'][:16].replace('T', ' ')} | Score Oráculo: **{b['oracle_score']:.1f}/100**")
            with st_col2:
                st.markdown(f"Stake: **{format_money(b['stake'], selected_curr)}**")
            with st_col3:
                status_color = "green" if b["status"] == "Ganada" else ("red" if b["status"] == "Perdida" else "orange")
                st.markdown(f":{status_color}[**{b['status']}**]")
            with st_col4:
                new_st = st.selectbox("Actualizar:", ["Pendiente", "Ganada", "Perdida", "Anulada"], index=["Pendiente", "Ganada", "Perdida", "Anulada"].index(b["status"]), key=f"status_sel_{b_id}")
                if new_st != b["status"]:
                    update_bet_status(b_id, new_st)
                    st.rerun()
    else:
        st.info("Aún no has registrado apuestas en SQLite. Agrega oportunidades desde el Escáner o desde 'Crear Apuesta'.")
