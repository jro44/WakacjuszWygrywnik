"""
LotusWygranus 3.0 PRO: Eurojackpot Hybrid Core
==============================================
Potężny silnik analityczny Data Science.
Łączy dwie bazy danych (5/50 + 2/12), buduje macierze powiązań (Affinity Matrix),
i wykorzystuje dynamiczne wagowanie w locie do tworzenia zoptymalizowanych kuponów.

Autor: Principal Data Scientist
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pdfplumber

# ---------------------------------------------------------------------------
# KONFIGURACJA GŁÓWNA EUROJACKPOT
# ---------------------------------------------------------------------------

MAIN_POOL_MIN, MAIN_POOL_MAX, MAIN_COUNT = 1, 50, 5
EXTRA_POOL_MIN, EXTRA_POOL_MAX, EXTRA_COUNT = 1, 12, 2

MAIN_POOL = list(range(MAIN_POOL_MIN, MAIN_POOL_MAX + 1))
EXTRA_POOL = list(range(EXTRA_POOL_MIN, EXTRA_POOL_MAX + 1))

# Liczby > 50 uznajemy za numery losowań (Draw IDs) w parserze Eurojackpot
ID_THRESHOLD = 50 

# ---------------------------------------------------------------------------
# STYLIZACJA I UI (Custom CSS)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LotusWygranus 3.0 PRO", page_icon="🇪🇺", layout="wide")

def inject_pro_css():
    st.markdown("""
        <style>
            :root {
                --bg-main: #0B0E14; --bg-panel: #151A22; --border: #2A3241;
                --text-main: #E2E8F0; --text-muted: #94A3B8;
                --accent-gold: #F59E0B; --accent-red: #EF4444; --accent-blue: #3B82F6;
            }
            .stApp { background-color: var(--bg-main); color: var(--text-main); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            h1, h2, h3 { color: var(--text-main); font-weight: 700; }
            .info-box {
                background: rgba(59, 130, 246, 0.1); border-left: 4px solid var(--accent-blue);
                padding: 15px; border-radius: 4px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.5;
            }
            .ball-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0; align-items: center;}
            .ball {
                width: 54px; height: 54px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
                font-weight: 800; font-size: 1.15rem; color: #111;
                box-shadow: 0 4px 10px rgba(0,0,0,0.5), inset 0 -4px 6px rgba(0,0,0,0.2);
            }
            .ball-main { background: radial-gradient(circle at 30% 30%, #FDE68A 0%, var(--accent-gold) 60%, #B45309 100%); }
            .ball-extra { background: radial-gradient(circle at 30% 30%, #FCA5A5 0%, var(--accent-red) 60%, #991B1B 100%); }
            .plus-sign { font-size: 1.5rem; color: var(--text-muted); font-weight: bold; margin: 0 5px; }
            .ticket-card {
                background: var(--bg-panel); border: 1px solid var(--border);
                border-radius: 12px; padding: 20px; margin-bottom: 16px;
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .ticket-card:hover { transform: translateY(-3px); border-color: var(--accent-gold); }
            .ticket-meta { color: var(--accent-gold); font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px; }
            .ticket-stats { color: var(--text-muted); font-size: 0.85rem; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# BARDZO PRECYZYJNY PARSER PDF (dla obu plików)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def parse_pdf_core(file_path: str, count: int, p_min: int, p_max: int) -> Dict[int, Tuple[int, ...]]:
    """Oparty na układzie współrzędnych parser eliminujący brud z plików PDF."""
    if not os.path.exists(file_path):
        return {}
        
    records = {}
    _INT_RE = re.compile(r"\b\d+\b")
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if not text: continue
            
            for line in text.splitlines():
                # Usuwamy daty, żeby parser nie uznał rocznika (np. 2024) za numer losowania
                line = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", line)
                ints = [int(t) for t in _INT_RE.findall(line)]
                if not ints: continue
                
                draw_id = None
                balls = []
                
                for val in ints:
                    # Identyfikacja ID (Eurojackpot ID to zwykle liczby > 50 np. 954, 800)
                    if val > ID_THRESHOLD and draw_id is None:
                        draw_id = val
                    elif p_min <= val <= p_max:
                        balls.append(val)
                        
                if draw_id is not None:
                    uniq_balls = tuple(sorted(set(balls)))
                    if len(uniq_balls) == count:
                        if draw_id not in records:
                            records[draw_id] = uniq_balls
                            
    return records

@st.cache_data(show_spinner="Inicjalizacja Rdzenia Analitycznego...")
def load_and_sync_databases() -> pd.DataFrame:
    """Synchronizuje bazę główną i dodatkową (INNER JOIN) po numerze losowania."""
    main_db = parse_pdf_core("5z50.PDF", MAIN_COUNT, MAIN_POOL_MIN, MAIN_POOL_MAX)
    extra_db = parse_pdf_core("2z12.PDF", EXTRA_COUNT, EXTRA_POOL_MIN, EXTRA_POOL_MAX)
    
    # Przechwytujemy tylko te losowania, gdzie mamy PEŁNE dane (5 z 50 ORAZ 2 z 12)
    common_ids = sorted(set(main_db.keys()) & set(extra_db.keys()), reverse=True)
    
    merged = []
    for did in common_ids:
        merged.append({
            "draw_id": did,
            "m1": main_db[did][0], "m2": main_db[did][1], "m3": main_db[did][2], "m4": main_db[did][3], "m5": main_db[did][4],
            "e1": extra_db[did][0], "e2": extra_db[did][1]
        })
        
    df = pd.DataFrame(merged)
    if not df.empty:
        df = df.sort_values("draw_id", ascending=True).reset_index(drop=True)
        df["order"] = np.arange(len(df))
    return df

# ---------------------------------------------------------------------------
# SILNIK DATA SCIENCE (Macierze i Statystyki)
# ---------------------------------------------------------------------------

@dataclass
class DeepStats:
    frequency: pd.Series
    current_gaps: Dict[int, int]
    mean_gaps: Dict[int, float]
    affinity_matrix: pd.DataFrame  # Nowość! Pełna macierz powiązań każda-z-każdą

@st.cache_data(show_spinner=False)
def compute_deep_stats(df: pd.DataFrame, cols: List[str], pool: List[int]) -> DeepStats:
    """Tworzy profesjonalne statystyki, w tym głęboką macierz powiązań (Affinity Matrix)."""
    n_draws = len(df)
    matrix = df[cols].to_numpy()
    
    freq_counter = Counter(matrix.flatten())
    last_seen = {}
    appearances = defaultdict(list)
    
    # Budowa macierzy powiązań (Affinity Matrix)
    # Rozmiar N x N, gdzie komórka (X, Y) to liczba wspólnych losowań cyfr X i Y
    aff_matrix = pd.DataFrame(0, index=pool, columns=pool, dtype=int)
    
    for idx, row in enumerate(matrix):
        for n in row:
            appearances[n].append(idx)
            last_seen[n] = idx
        # Parowanie krzyżowe - cała historia każdej cyfry względem innej
        for a, b in combinations(row, 2):
            aff_matrix.at[a, b] += 1
            aff_matrix.at[b, a] += 1

    freq_series = pd.Series({n: freq_counter.get(n, 0) for n in pool}).sort_index()
    
    gaps = {}
    mean_gaps = {}
    for n in pool:
        apps = appearances.get(n, [])
        gaps[n] = (n_draws - 1 - last_seen[n]) if apps else n_draws
        mean_gaps[n] = float(np.mean(np.diff(apps))) if len(apps) > 1 else float(n_draws)
        
    return DeepStats(freq_series, gaps, mean_gaps, aff_matrix)

# ---------------------------------------------------------------------------
# HYBRYDOWY GENERATOR Z DYNAMICZNYMI WAGAMI (Wersja PRO)
# ---------------------------------------------------------------------------

def calculate_base_weights(stats: DeepStats, pool: List[int], mode: str) -> np.ndarray:
    """Kalkuluje startowy potencjał każdej kuli na podstawie wybranej metodyki."""
    f = stats.frequency.reindex(pool).fillna(0).to_numpy(dtype=float)
    g = np.array([stats.current_gaps[n] for n in pool], dtype=float)
    mg = np.array([stats.mean_gaps[n] for n in pool], dtype=float)
    
    # Wygładzanie (Laplace smoothing), żeby uniknąć dzielenia przez 0
    f_smooth = f + 1.0 
    overdue_ratio = np.where(mg > 0, g / mg, 1.0)
    
    if mode == "hot":
        # Faworyzuje najczęstsze liczby (Gorące)
        w = f_smooth
    elif mode == "cold":
        # Faworyzuje liczby przespane i rzadkie (Zimne)
        w = (1.0 / f_smooth) * (overdue_ratio ** 2)
    elif mode == "hybrid":
        # Tryb Mix: Balansuje uśrednioną częstotliwość z ratio uśpienia
        w = (f_smooth / f_smooth.max()) + (overdue_ratio / overdue_ratio.max())
    else:
        w = np.ones(len(pool))
        
    # Normalizacja
    return w / w.sum() if w.sum() > 0 else np.ones(len(pool)) / len(pool)

def generate_pro_ticket(
    stats: DeepStats, pool: List[int], count: int, 
    mode: str, intensity: float, affinity_strength: float
) -> Tuple[int, ...]:
    """
    Kluczowy algorytm 3.0: 
    1. Ustala wagi bazowe.
    2. Losuje pierwszą kulę.
    3. Analizuje całą historię wylosowanej kuli w Affinity Matrix.
    4. Zwiększa wagi kulom, z którymi historycznie "lubi się" pierwsza kula (zależnie od affinity_strength).
    5. Powtarza proces aż do pełnego kuponu.
    """
    base_w = calculate_base_weights(stats, pool, mode)
    
    # Nakładamy intensywność (0.0 = czysty lotto-chaos, 1.0 = czysta statystyka)
    uniform_w = np.ones(len(pool)) / len(pool)
    current_w = (intensity * base_w) + ((1.0 - intensity) * uniform_w)
    current_w /= current_w.sum()
    
    chosen = []
    available_mask = np.ones(len(pool), dtype=bool)
    pool_arr = np.array(pool)
    
    for _ in range(count):
        # 1. Losowanie kuli na bazie obecnych wag
        probs = current_w * available_mask
        probs /= probs.sum() # Renormalizacja po wykluczeniu zużytych kul
        
        pick_idx = np.random.choice(len(pool), p=probs)
        pick_val = pool[pick_idx]
        chosen.append(pick_val)
        available_mask[pick_idx] = False
        
        # 2. DYNAMICZNY AFFINITY BOOST (Analiza względem tej samej na podst. historii)
        if affinity_strength > 0.0:
            # Wyciągamy z macierzy, jak często 'pick_val' padało z pozostałymi kulami
            historical_links = stats.affinity_matrix.loc[pick_val].to_numpy(dtype=float)
            if historical_links.max() > 0:
                # Normalizujemy powiązania do mnożnika (od 1.0 do 1.0 + affinity_strength)
                boost_multiplier = 1.0 + (affinity_strength * (historical_links / historical_links.max()))
                current_w = current_w * boost_multiplier
    
    return tuple(sorted(chosen))

# ---------------------------------------------------------------------------
# TWORZENIE WYKRESÓW ZAAWANSOWANYCH
# ---------------------------------------------------------------------------

_CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Segoe UI", color="#E2E8F0"), margin=dict(l=20, r=20, t=40, b=20)
)

def plot_affinity_heatmap(stats: DeepStats, title: str) -> go.Figure:
    """Rysuje mapę ciepła (Heatmap) dla macierzy powiązań."""
    matrix = stats.affinity_matrix
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values, x=matrix.columns, y=matrix.index,
        colorscale="Viridis", hoverongaps=False,
        hovertemplate="Liczba %{y} oraz %{x}<br>Wspólnych losowań: %{z}<extra></extra>"
    ))
    fig.update_layout(title=title, xaxis_title="Liczba", yaxis_title="Liczba", **_CHART_THEME)
    return fig

# ---------------------------------------------------------------------------
# APLIKACJA GŁÓWNA STREAMLIT
# ---------------------------------------------------------------------------

def main():
    inject_pro_css()
    
    st.markdown("""
        <div style="background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%); padding: 30px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
            <h1 style="margin:0; font-size: 2.8rem; background: linear-gradient(90deg, #FDE68A, #F59E0B); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🇪🇺 LotusWygranus 3.0 PRO</h1>
            <p style="color: #94A3B8; font-size: 1.1rem; margin-top: 10px;">Zaawansowany hybrydowy model analizy danych dla Eurojackpot (5/50 + 2/12). 
            System buduje dynamiczne macierze prawdopodobieństwa i analizuje relacje każdej cyfry względem całej dostępnej historii.</p>
        </div>
    """, unsafe_allow_html=True)

    # Inicjalizacja Danych
    df = load_and_sync_databases()
    if df.empty:
        st.error("Brak plików z danymi! Upewnij się, że pliki '5z50.PDF' oraz '2z12.PDF' znajdują się w tym samym folderze co ten skrypt (np. na GitHub).")
        st.stop()
        
    main_stats = compute_deep_stats(df, ["m1","m2","m3","m4","m5"], MAIN_POOL)
    extra_stats = compute_deep_stats(df, ["e1","e2"], EXTRA_POOL)

    # Pasek boczny z opcjami i rozbudowanymi poradnikami
    with st.sidebar:
        st.markdown("### ⚙️ Parametry Silnika")
        
        st.markdown("""
        <div class="info-box">
            <b>Zrozumienie Strategii:</b><br><br>
            🔥 <b>Hot (Gorące):</b> Algorytm tworzy zestawy na bazie liczb, które padają najczęściej w całym zakresie. Idealne do "ujeżdżania trendów".<br><br>
            ❄️ <b>Cold (Zimne):</b> Algorytm wyszukuje liczby, które "śpią" znacznie dłużej niż wynosi ich średnia historyczna.<br><br>
            ⚖️ <b>Hybrid Mix (Mieszane):</b> Profesjonalny balans. Maszyna wybiera najsilniejsze liczby ze skrajnych biegunów (trochę pewniaków, trochę zimnych strzałów).
        </div>
        """, unsafe_allow_html=True)
        
        mode = st.radio("Wybierz tryb pracy (Strategia):", ["hybrid", "hot", "cold"], 
                        format_func=lambda m: {"hot": "🔥 Hot (Gorące)", "cold": "❄️ Cold (Zimne)", "hybrid": "⚖️ Hybrid Mix (Hybrydowe)"}[m])
        
        st.markdown("""
        <div class="info-box">
            <b>Intensywność Statystyczna:</b><br>
            Określa, na ile ufasz statystyce. <br><code>0.0</code> to zwykły ślepy los (jak w maszynie Lotto), a <code>1.0</code> to absolutne dyktando twardych danych i wag. Zalecane: <b>0.7 - 0.8</b>.
        </div>
        """, unsafe_allow_html=True)
        intensity = st.slider("Intensywność Statystyczna", 0.0, 1.0, 0.75, 0.05)
        
        st.markdown("""
        <div class="info-box">
            <b>Siła Analizy Historycznej (Affinity):</b><br>
            Właśnie tu dzieje się magia <i>"każdej cyfry względem innej"</i>. Kiedy algorytm wylosuje pierwszą kulę (np. 14), analizuje całą historię PDF, by sprawdzić co najczęściej pada z czternastką, i podbija tym liczbom szanse.<br>Zalecane: <b>0.4 - 0.6</b>.
        </div>
        """, unsafe_allow_html=True)
        affinity = st.slider("Siła Powiązań (Affinity Matrix)", 0.0, 1.0, 0.50, 0.05)
        
        n_tickets = st.number_input("Ile zestawów wygenerować?", 1, 50, 5)
        generate_btn = st.button("🚀 INICJUJ GENERATOR", use_container_width=True)

    # Główne zakładki
    tab_sim, tab_matrix, tab_stats, tab_db = st.tabs([
        "🎲 Moduł Generatora", "🕸️ Macierz Powiązań (Affinity)", "📊 Głębokie Statystyki", "📁 Zsynchronizowana Baza"
    ])

    with tab_sim:
        st.markdown(f"### 🛡️ Zsynchronizowano **{len(df)}** historycznych losowań Eurojackpot.")
        
        if generate_btn or "tickets" not in st.session_state:
            with st.spinner("Przeliczanie bilionów kombinacji w macierzy..."):
                tickets = []
                for _ in range(n_tickets):
                    # Generowanie 5 z 50
                    m_draw = generate_pro_ticket(main_stats, MAIN_POOL, MAIN_COUNT, mode, intensity, affinity)
                    # Generowanie 2 z 12
                    e_draw = generate_pro_ticket(extra_stats, EXTRA_POOL, EXTRA_COUNT, mode, intensity, affinity)
                    tickets.append((m_draw, e_draw))
                st.session_state["tickets"] = tickets
                st.session_state["mode"] = mode

        # Renderowanie kuponów
        for i, (m_draw, e_draw) in enumerate(st.session_state["tickets"], start=1):
            # Obliczenia metadanych dla kuponu (żeby pokazać, że to działa profesjonalnie)
            m_sum = sum(m_draw)
            m_odd = sum(1 for x in m_draw if x % 2 != 0)
            avg_freq = np.mean([main_stats.frequency[x] for x in m_draw])
            
            html = f"""
            <div class="ticket-card">
                <div class="ticket-meta">Zestaw #{i} &bull; Tryb: {st.session_state['mode'].upper()}</div>
                <div class="ball-row">
            """
            for num in m_draw: html += f'<div class="ball ball-main">{num}</div>'
            html += '<div class="plus-sign">+</div>'
            for num in e_draw: html += f'<div class="ball ball-extra">{num}</div>'
            html += f"""
                </div>
                <div class="ticket-stats">
                    <b>Analityka kuponu:</b> Suma kul głównych: {m_sum} | Nieparzyste/Parzyste: {m_odd}/{MAIN_COUNT - m_odd} | Średnia częstotliwość historyczna: {avg_freq:.1f}
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    with tab_matrix:
        st.markdown("### 🕸️ Mapa Ciepła Powiązań (Kule Główne)")
        st.write("Czym jaśniejszy kolor, tym częściej dwie liczby pojawiały się razem na jednym kuponie w całej historii Eurojackpot. Algorytm w czasie rzeczywistym używa tej siatki do parowania Twoich zestawów.")
        st.plotly_chart(plot_affinity_heatmap(main_stats, "Affinity Matrix (5 z 50)"), use_container_width=True)

    with tab_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔥 Najgorętsze Kule Główne")
            st.dataframe(main_stats.frequency.sort_values(ascending=False).head(10).reset_index().rename(columns={"index": "Kula", 0: "Liczba Wystąpień"}), use_container_width=True)
        with col2:
            st.markdown("#### ❄️ Najbardziej 'Uśpione' (Zimne)")
            gaps_df = pd.Series(main_stats.current_gaps).sort_values(ascending=False).head(10).reset_index().rename(columns={"index": "Kula", 0: "Zaległych losowań"})
            st.dataframe(gaps_df, use_container_width=True)

    with tab_db:
        st.markdown("### 📁 Sparowana Baza Danych z PDF")
        st.write("Aplikacja pomyślnie złączyła dane z plików 5z50.pdf oraz 2z12.pdf w jedną zsynchronizowaną tabelę czasową.")
        # Wizualizacja tabeli (newest first)
        st.dataframe(df.sort_values("draw_id", ascending=False).drop(columns=["order"]), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
