"""
LotusWygranus 4.0 ULTIMATE: Eurojackpot AI & Markov Sequences
=============================================================
Profesjonalny system predykcyjny analizujący "szlaki" maszyny losującej
od najstarszego do najnowszego losowania za pomocą Łańcuchów Markowa
oraz twardej analizy powiązań krzyżowych.

Autor: Principal Data Scientist & Software Architect
"""

from __future__ import annotations

import io
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pdfplumber

# ---------------------------------------------------------------------------
# KONFIGURACJA EUROJACKPOT
# ---------------------------------------------------------------------------

MAIN_POOL_MIN, MAIN_POOL_MAX, MAIN_COUNT = 1, 50, 5
EXTRA_POOL_MIN, EXTRA_POOL_MAX, EXTRA_COUNT = 1, 12, 2

MAIN_POOL = list(range(MAIN_POOL_MIN, MAIN_POOL_MAX + 1))
EXTRA_POOL = list(range(EXTRA_POOL_MIN, EXTRA_POOL_MAX + 1))

# Próg dla Draw ID (żeby odsiać kule od numerów losowań Eurojackpot)
ID_THRESHOLD = 50 

# ---------------------------------------------------------------------------
# STYLIZACJA INTERFEJSU
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LotusWygranus 4.0", page_icon="🔮", layout="wide")

def inject_ultimate_css():
    st.markdown("""
        <style>
            :root {
                --bg-main: #0a0a0a; --bg-panel: #111111; --border: #333333;
                --text-main: #f5f5f5; --text-muted: #888888;
                --accent-gold: #D4AF37; --accent-crimson: #DC143C; --accent-neon: #00FFCC;
            }
            .stApp { background-color: var(--bg-main); color: var(--text-main); font-family: 'Inter', sans-serif; }
            h1, h2, h3 { color: var(--text-main); font-weight: 800; letter-spacing: -0.5px; }
            
            /* Boxy informacyjne */
            .expert-box {
                background: linear-gradient(145deg, rgba(212, 175, 55, 0.1), rgba(0,0,0,0));
                border-left: 4px solid var(--accent-gold);
                padding: 16px; border-radius: 4px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.6;
            }
            .expert-title { font-weight: 800; color: var(--accent-gold); margin-bottom: 8px; font-size: 1.1rem; }
            
            /* Kule losujące */
            .ball-container { display: flex; gap: 10px; flex-wrap: wrap; margin: 15px 0; align-items: center; }
            .ball {
                width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
                font-weight: 800; font-size: 1.2rem; color: #000;
                box-shadow: 0 4px 15px rgba(0,0,0,0.6), inset 0 -4px 8px rgba(0,0,0,0.3);
                border: 2px solid rgba(255,255,255,0.2);
            }
            .ball-main { background: radial-gradient(circle at 35% 35%, #fffde7 0%, var(--accent-gold) 50%, #8b6914 100%); }
            .ball-extra { background: radial-gradient(circle at 35% 35%, #ffebee 0%, var(--accent-crimson) 50%, #8b0000 100%); }
            .plus-sign { font-size: 1.8rem; color: var(--text-muted); font-weight: 900; margin: 0 10px; }
            
            /* Kupony */
            .ticket-card {
                background: var(--bg-panel); border: 1px solid var(--border);
                border-radius: 8px; padding: 22px; margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                transition: transform 0.2s, border-color 0.2s;
            }
            .ticket-card:hover { transform: translateY(-2px); border-color: var(--accent-neon); }
            .ticket-header { color: var(--accent-neon); font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 12px; }
            .ticket-footer { color: var(--text-muted); font-size: 0.85rem; margin-top: 15px; padding-top: 15px; border-top: 1px solid var(--border); }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PRECYZYJNY PARSER PDF
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def parse_pdf_core(file_path: str, count: int, p_min: int, p_max: int) -> Dict[int, Tuple[int, ...]]:
    if not os.path.exists(file_path):
        return {}
        
    records = {}
    _INT_RE = re.compile(r"\b\d+\b")
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if not text: continue
            
            for line in text.splitlines():
                # Czyszczenie dat
                line = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", line)
                ints = [int(t) for t in _INT_RE.findall(line)]
                if not ints: continue
                
                draw_id, balls = None, []
                for val in ints:
                    if val > ID_THRESHOLD and draw_id is None: draw_id = val
                    elif p_min <= val <= p_max: balls.append(val)
                        
                if draw_id is not None:
                    uniq_balls = tuple(sorted(set(balls)))
                    if len(uniq_balls) == count:
                        if draw_id not in records:
                            records[draw_id] = uniq_balls
    return records

@st.cache_data(show_spinner="Synchronizacja Baz Danych (Łączenie strumieni)...")
def load_and_sync_databases() -> pd.DataFrame:
    main_db = parse_pdf_core("5z50.PDF", MAIN_COUNT, MAIN_POOL_MIN, MAIN_POOL_MAX)
    extra_db = parse_pdf_core("2z12.PDF", EXTRA_COUNT, EXTRA_POOL_MIN, EXTRA_POOL_MAX)
    
    common_ids = sorted(set(main_db.keys()) & set(extra_db.keys())) # Sortujemy rosnąco (najstarsze pierwsze)
    
    merged = []
    for did in common_ids:
        merged.append({
            "draw_id": did,
            "m1": main_db[did][0], "m2": main_db[did][1], "m3": main_db[did][2], "m4": main_db[did][3], "m5": main_db[did][4],
            "e1": extra_db[did][0], "e2": extra_db[did][1]
        })
        
    df = pd.DataFrame(merged)
    if not df.empty:
        # Twarde sortowanie chronologiczne od najstarszego
        df = df.sort_values("draw_id", ascending=True).reset_index(drop=True)
        df["order"] = np.arange(len(df))
    return df

# ---------------------------------------------------------------------------
# SILNIK DATA SCIENCE (Szlak Markowa & Affinity Matrix)
# ---------------------------------------------------------------------------

@dataclass
class UltimateStats:
    frequency: pd.Series
    current_gaps: Dict[int, int]
    mean_gaps: Dict[int, float]
    affinity_matrix: pd.DataFrame     # Wystąpienia w TMY SAMYM losowaniu
    transition_matrix: pd.DataFrame   # Przejścia z losowania T do T+1 (Szlak Markowa)
    last_drawn: Tuple[int, ...]       # Ostatnio wylosowane kule

@st.cache_data(show_spinner=False)
def compute_ultimate_stats(df: pd.DataFrame, cols: List[str], pool: List[int]) -> UltimateStats:
    n_draws = len(df)
    matrix = df[cols].to_numpy()
    
    freq_counter = Counter()
    last_seen = {}
    appearances = defaultdict(list)
    
    aff_matrix = pd.DataFrame(0.0, index=pool, columns=pool, dtype=float)
    trans_matrix = pd.DataFrame(0.0, index=pool, columns=pool, dtype=float)
    
    # Przechodzimy po losowaniach od najstarszego do najnowszego
    for i in range(n_draws):
        row = matrix[i]
        freq_counter.update(row)
        
        for n in row:
            appearances[n].append(i)
            last_seen[n] = i
            
        # Budowa Affinity Matrix (Kto lubi kogo w tym samym losowaniu)
        for a, b in combinations(row, 2):
            aff_matrix.at[a, b] += 1.0
            aff_matrix.at[b, a] += 1.0
            
        # Budowa Transition Matrix (Szlak Markowa: co wypadło w losowaniu i+1 względem i)
        if i < n_draws - 1:
            next_row = matrix[i+1]
            for current_ball in row:
                for next_ball in next_row:
                    trans_matrix.at[current_ball, next_ball] += 1.0

    freq_series = pd.Series({n: freq_counter.get(n, 0) for n in pool}).sort_index()
    
    gaps, mean_gaps = {}, {}
    for n in pool:
        apps = appearances.get(n, [])
        gaps[n] = (n_draws - 1 - last_seen[n]) if apps else n_draws
        mean_gaps[n] = float(np.mean(np.diff(apps))) if len(apps) > 1 else float(n_draws)
        
    last_drawn = tuple(matrix[-1]) if n_draws > 0 else tuple()
        
    return UltimateStats(freq_series, gaps, mean_gaps, aff_matrix, trans_matrix, last_drawn)

# ---------------------------------------------------------------------------
# GENERATOR Z LOKOMOTYWĄ MARKOWA
# ---------------------------------------------------------------------------

def _normalize(weights: np.ndarray) -> np.ndarray:
    w = np.copy(weights)
    w[w < 0] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def calculate_base_weights(stats: UltimateStats, pool: List[int], mode: str) -> np.ndarray:
    f = stats.frequency.reindex(pool).fillna(0).to_numpy(dtype=float) + 1.0
    g = np.array([stats.current_gaps[n] for n in pool], dtype=float)
    mg = np.array([stats.mean_gaps[n] for n in pool], dtype=float)
    
    overdue_ratio = np.where(mg > 0, g / mg, 1.0)
    
    if mode == "hot":
        w = f
    elif mode == "cold":
        w = (1.0 / f) * (overdue_ratio ** 2)
    elif mode == "hybrid":
        w = (f / f.max()) + (overdue_ratio / overdue_ratio.max())
    elif mode == "markov":
        # Szlak Markowa: Sprawdzamy co wypadło ostatnio i sumujemy prawdopodobieństwa przejść
        w = np.zeros(len(pool))
        for last_ball in stats.last_drawn:
            transitions = stats.transition_matrix.loc[last_ball].to_numpy()
            w += transitions
        w += 0.1 # Smoothing, żeby kule z zerowym przejściem miały ułamek szansy
    else:
        w = np.ones(len(pool))
        
    return _normalize(w)

def generate_ultimate_ticket(
    stats: UltimateStats, pool: List[int], count: int, 
    mode: str, intensity: float, affinity_strength: float
) -> Tuple[int, ...]:
    
    base_w = calculate_base_weights(stats, pool, mode)
    uniform_w = np.ones(len(pool)) / len(pool)
    
    # Aplikacja intensywności (balans między czystą losowością a wagami modelu)
    current_w = _normalize((intensity * base_w) + ((1.0 - intensity) * uniform_w))
    
    chosen = []
    available_mask = np.ones(len(pool), dtype=bool)
    
    for _ in range(count):
        probs = _normalize(current_w * available_mask)
        pick_idx = np.random.choice(len(pool), p=probs)
        pick_val = pool[pick_idx]
        chosen.append(pick_val)
        available_mask[pick_idx] = False
        
        # Affinity Boost (Wpływ kul między sobą na tym samym kuponie)
        if affinity_strength > 0.0:
            links = stats.affinity_matrix.loc[pick_val].to_numpy()
            if links.max() > 0:
                boost = 1.0 + (affinity_strength * (links / links.max()))
                current_w = _normalize(current_w * boost)
    
    return tuple(sorted(chosen))

# ---------------------------------------------------------------------------
# WYKRESY
# ---------------------------------------------------------------------------

_CHART_THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#f5f5f5"), margin=dict(l=10, r=10, t=40, b=10))

def plot_transition_heatmap(stats: UltimateStats, title: str) -> go.Figure:
    matrix = stats.transition_matrix
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values, x=matrix.columns, y=matrix.index,
        colorscale="Inferno", hoverongaps=False,
        hovertemplate="Kula T: %{y} ➜ Kula T+1: %{x}<br>Powtórzeń szlaku: %{z}<extra></extra>"
    ))
    fig.update_layout(title=title, xaxis_title="Wylosowana zaraz PÓŹNIEJ (T+1)", yaxis_title="Wylosowana kula (T)", **_CHART_THEME)
    return fig

# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------

def main():
    inject_ultimate_css()
    
    st.markdown("""
        <div style="background: #111; padding: 25px; border-radius: 6px; border: 1px solid #333; margin-bottom: 25px;">
            <h1 style="margin:0; font-size: 2.5rem;"><span style="color: #00FFCC;">⬡</span> LotusWygranus 4.0 ULTIMATE</h1>
            <p style="color: #888; font-size: 1.1rem; margin-top: 5px;">Rdzeń Analizy Wektorowej & Łańcuchy Markowa dla Eurojackpot.</p>
        </div>
    """, unsafe_allow_html=True)

    df = load_and_sync_databases()
    if df.empty:
        st.error("Brak plików '5z50.PDF' i '2z12.PDF' w repozytorium GitHub.")
        st.stop()
        
    main_stats = compute_ultimate_stats(df, ["m1","m2","m3","m4","m5"], MAIN_POOL)
    extra_stats = compute_ultimate_stats(df, ["e1","e2"], EXTRA_POOL)

    with st.sidebar:
        st.markdown("### 🎛️ PANEL KONTROLNY")
        
        st.markdown("""
        <div class="expert-box">
            <div class="expert-title">Tryb "Szlak Markowa" (NOWOŚĆ)</div>
            Wykorzystuje pamięć fizyczną maszyny losującej. Algorytm sprawdza, jakie kule wypadły w <b>ostatnim losowaniu w pliku PDF</b> i oblicza, co matematycznie najczęściej pada po nich jako "skok" maszyny. Wzorowanie się na śladach od najstarszego do najnowszego.
        </div>
        """, unsafe_allow_html=True)
        
        mode = st.radio("Metodyka wyboru wag:", ["markov", "hybrid", "hot", "cold"], 
                        format_func=lambda m: {
                            "markov": "⬡ Szlak Markowa (Skoki Maszyny)", 
                            "hot": "🔥 Gorące (Trendy)", 
                            "cold": "❄️ Zimne (Na przełamanie)", 
                            "hybrid": "⚖️ Hybryda (Balans)"
                        }[m])
        
        intensity = st.slider("Intensywność Ufności (Wagi algorytmu)", 0.0, 1.0, 0.80, 0.05, help="1.0 = Ślepe podążanie za statystyką (ryzyko powtarzalności). 0.0 = Lotto na ślepo. Zalecane: 0.7-0.8")
        affinity = st.slider("Siła Powiązań (Pary na kuponie)", 0.0, 1.0, 0.50, 0.05, help="Dobiera na jednym kuponie kule, które historycznie mocno ze sobą współpracują.")
        n_tickets = st.number_input("Generowane zakłady", 1, 50, 6)
        generate_btn = st.button("🚀 INICJALIZACJA PREDYKCJI", use_container_width=True)

    tab_sim, tab_markov, tab_stats = st.tabs(["🔮 Predykcje", "⬡ Mapy Przejść (Markow)", "📊 Statystyki bazy"])

    with tab_sim:
        st.write(f"Zsynchronizowano **{len(df)}** losowań Eurojackpot. Ostatnie zanotowane losowanie [ID: {df.iloc[-1]['draw_id']}]")
        
        if mode == "markov":
            st.markdown(f"**Analizuję skoki maszyny bazując na ostatnim losowaniu:**")
            st.markdown(f"Kule Główne: `{main_stats.last_drawn}` | Euro: `{extra_stats.last_drawn}`")
        
        if generate_btn or "tickets" not in st.session_state:
            with st.spinner("Kompilacja wag, analiza wektorowa przejść..."):
                tickets = []
                for _ in range(n_tickets):
                    m_draw = generate_ultimate_ticket(main_stats, MAIN_POOL, MAIN_COUNT, mode, intensity, affinity)
                    e_draw = generate_ultimate_ticket(extra_stats, EXTRA_POOL, EXTRA_COUNT, mode, intensity, affinity)
                    tickets.append((m_draw, e_draw))
                st.session_state["tickets"] = tickets
                st.session_state["mode"] = mode

        for i, (m_draw, e_draw) in enumerate(st.session_state["tickets"], start=1):
            html = f"""
            <div class="ticket-card">
                <div class="ticket-header">Zakład #{i} &bull; Profil: {st.session_state['mode'].upper()}</div>
                <div class="ball-container">
            """
            for num in m_draw: html += f'<div class="ball ball-main">{num}</div>'
            html += '<div class="plus-sign">+</div>'
            for num in e_draw: html += f'<div class="ball ball-extra">{num}</div>'
            html += f"""
                </div>
                <div class="ticket-footer">Algorytm wziął pod uwagę siłę przejść między najstarszym a najnowszym losowaniem.</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    with tab_markov:
        st.markdown("### ⬡ Szlaki Maszyny: Macierz Przejść (Transition Matrix)")
        st.write("Wykres pokazuje historię przejść. Oś Y to kula z losowania nr *N*. Oś X to kula z losowania *N+1* (następnego).")
        st.plotly_chart(plot_transition_heatmap(main_stats, "Przejścia Łańcucha Markowa - Kule Główne"), use_container_width=True)

    with tab_stats:
        st.markdown("### Surowa baza odczytana z plików PDF")
        st.dataframe(df.sort_values("draw_id", ascending=False).drop(columns=["order"]), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
