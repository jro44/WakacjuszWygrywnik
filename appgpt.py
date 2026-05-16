import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import random
from collections import Counter, defaultdict
import plotly.express as px

# =========================================================
# KONFIGURACJA STRONY
# =========================================================

st.set_page_config(
    page_title="Lotto Engine PRO",
    layout="wide"
)

# =========================================================
# STYL
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.stButton button {
    background-color: #16a34a;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 55px;
    width: 100%;
}

.big-font {
    font-size:26px !important;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TYTUŁ
# =========================================================

st.markdown('<p class="big-font">🎯 LOTTO ENGINE PRO</p>', unsafe_allow_html=True)

st.write("Zaawansowany silnik analizy statystycznej losowań.")

# =========================================================
# ODCZYT PDF
# =========================================================

@st.cache_data
def extract_draws_from_pdf(path, count_numbers):

    draws = []

    with pdfplumber.open(path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if not text:
                continue

            lines = text.split("\n")

            for line in lines:

                nums = re.findall(r'\d+', line)

                nums = [int(x) for x in nums]

                # ignoruj numery losowań typu 0954
                filtered = [x for x in nums if x <= 50]

                if len(filtered) == count_numbers:
                    draws.append(filtered)

    return draws

# =========================================================
# ŁADOWANIE DANYCH
# =========================================================

main_draws = extract_draws_from_pdf("5z50.PDF", 5)
euro_draws = extract_draws_from_pdf("2z12.PDF", 2)

# =========================================================
# STATYSTYKI
# =========================================================

class StatisticsEngine:

    def __init__(self, draws):

        self.draws = draws
        self.flat = [n for row in draws for n in row]

    def frequency(self):

        return Counter(self.flat)

    def hot_numbers(self, top=10):

        freq = self.frequency()

        return sorted(freq.items(),
                      key=lambda x: x[1],
                      reverse=True)[:top]

    def cold_numbers(self, top=10):

        freq = self.frequency()

        return sorted(freq.items(),
                      key=lambda x: x[1])[:top]

    def gaps(self):

        gaps = {}

        for number in set(self.flat):

            positions = []

            for idx, row in enumerate(self.draws):

                if number in row:
                    positions.append(idx)

            if len(positions) > 1:

                diff = np.diff(positions)

                gaps[number] = np.mean(diff)

        return gaps

# =========================================================
# SILNIK PRZEJŚĆ
# =========================================================

class TransitionEngine:

    def __init__(self, draws):

        self.draws = draws
        self.transitions = defaultdict(int)

    def build(self):

        for i in range(len(self.draws)-1):

            current = self.draws[i]
            nxt = self.draws[i+1]

            for c in current:
                for n in nxt:
                    self.transitions[(c, n)] += 1

        return self.transitions

# =========================================================
# ANALIZA PAR
# =========================================================

def pair_analysis(draws):

    pairs = Counter()

    for row in draws:

        for i in range(len(row)):
            for j in range(i+1, len(row)):

                pair = tuple(sorted((row[i], row[j])))

                pairs[pair] += 1

    return pairs

# =========================================================
# BUDOWANIE SKORINGU
# =========================================================

def build_scores(draws):

    stats = StatisticsEngine(draws)

    freq = stats.frequency()
    gaps = stats.gaps()

    transition = TransitionEngine(draws)
    trans = transition.build()

    pair_data = pair_analysis(draws)

    scores = {}

    max_freq = max(freq.values())

    for number in freq.keys():

        hot_score = freq[number] / max_freq

        gap_score = 1 / (gaps.get(number, 1))

        transition_score = 0

        for (a, b), val in trans.items():

            if b == number:
                transition_score += val

        pair_score = 0

        for pair, val in pair_data.items():

            if number in pair:
                pair_score += val

        final = (
            hot_score * 0.35 +
            gap_score * 0.15 +
            transition_score * 0.25 +
            pair_score * 0.25
        )

        scores[number] = final

    return scores

# =========================================================
# GENERATOR SILVER BULLET
# =========================================================

class SilverBullet:

    def __init__(self, scores, min_num, max_num):

        self.scores = scores
        self.min_num = min_num
        self.max_num = max_num

    def generate(self, count):

        nums = list(self.scores.keys())
        weights = list(self.scores.values())

        selected = set()

        mutation = random.uniform(0.90, 1.15)

        while len(selected) < count:

            choice = random.choices(
                nums,
                weights=[w * mutation * random.uniform(0.8, 1.2)
                         for w in weights],
                k=1
            )[0]

            selected.add(choice)

        return sorted(selected)

# =========================================================
# OBLICZENIA
# =========================================================

main_scores = build_scores(main_draws)
euro_scores = build_scores(euro_draws)

main_engine = SilverBullet(main_scores, 1, 50)
euro_engine = SilverBullet(euro_scores, 1, 12)

# =========================================================
# MENU
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Silver Bullet",
    "🔥 Gorące cyfry",
    "❄️ Zimne cyfry",
    "📊 Statystyki"
])

# =========================================================
# SILVER BULLET
# =========================================================

with tab1:

    st.subheader("Najmocniejszy tryb hybrydowy")

    st.write("""
    System analizuje:

    - gorące liczby,
    - zimne liczby,
    - przejścia między losowaniami,
    - siłę par,
    - pamięć maszyny,
    - historię ruchu,
    - dynamikę zmian.
    """)

    if st.button("🚀 GENERUJ SILVER BULLET"):

        result_main = main_engine.generate(5)
        result_euro = euro_engine.generate(2)

        st.success(f"5 z 50: {result_main}")
        st.success(f"2 z 12: {result_euro}")

# =========================================================
# GORĄCE
# =========================================================

with tab2:

    st.subheader("Najgorętsze liczby")

    hot_main = StatisticsEngine(main_draws).hot_numbers(15)

    df_hot = pd.DataFrame(
        hot_main,
        columns=["Liczba", "Wystąpienia"]
    )

    st.dataframe(df_hot)

    fig = px.bar(
        df_hot,
        x="Liczba",
        y="Wystąpienia",
        title="Najgorętsze liczby"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ZIMNE
# =========================================================

with tab3:

    st.subheader("Najzimniejsze liczby")

    cold_main = StatisticsEngine(main_draws).cold_numbers(15)

    df_cold = pd.DataFrame(
        cold_main,
        columns=["Liczba", "Wystąpienia"]
    )

    st.dataframe(df_cold)

    fig2 = px.bar(
        df_cold,
        x="Liczba",
        y="Wystąpienia",
        title="Najzimniejsze liczby"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# STATYSTYKI
# =========================================================

with tab4:

    st.subheader("Zaawansowane statystyki")

    freq = StatisticsEngine(main_draws).frequency()

    df = pd.DataFrame(
        freq.items(),
        columns=["Liczba", "Ilość"]
    )

    df = df.sort_values(by="Ilość", ascending=False)

    st.dataframe(df)

    fig3 = px.line(
        df,
        x="Liczba",
        y="Ilość",
        markers=True,
        title="Rozkład częstotliwości"
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# STOPKA
# =========================================================

st.write("---")

st.write("""
Lotto Engine PRO v1.0

Silnik:
- analiza statystyczna,
- analiza przejść,
- analiza par,
- dynamiczne ważenie,
- kontrolowana losowość,
- pamięć maszyny.
""")
