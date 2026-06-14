"""
CineMatch — Film Recommendation System with SVD
Streamlit Frontend
"""

import os
import sys
import zipfile
import urllib.request
import warnings
import joblib
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Film Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #2a2a4a;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #2a2a4a;
}
.hero-banner h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: #e94560;
    margin: 0;
    letter-spacing: -1px;
}
.hero-banner p {
    color: #a0aec0;
    font-size: 1.05rem;
    margin: 0.4rem 0 0;
}

/* Metric cards */
.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e94560;
}
.metric-card .label {
    color: #a0aec0;
    font-size: 0.85rem;
    margin-top: 4px;
}

/* Recommendation card */
.rec-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: #e94560; }
.rec-rank {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e94560;
    min-width: 36px;
}
.rec-title { font-weight: 600; color: #e2e8f0; font-size: 1rem; }
.rec-genre { color: #718096; font-size: 0.82rem; margin-top: 2px; }
.rec-rating {
    margin-left: auto;
    text-align: right;
}
.rec-stars { color: #f6c90e; font-size: 1rem; }
.rec-score { color: #a0aec0; font-size: 0.8rem; }

/* Section title */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #e94560;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Info box */
.info-box {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #a0aec0;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* Streamlit elements overrides */
div[data-testid="stNumberInput"] input { background: #1a1a2e !important; color: #e2e8f0 !important; }
div[data-testid="stSlider"] { color: #e2e8f0; }
label { color: #a0aec0 !important; }
h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH    = "ml-latest-small.zip"
DATA_DIR    = "ml-latest-small"
MODEL_PATH  = "cinématch_svd_model.pkl"
MIN_RATINGS = 20


# ─── Data & Model Loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_dataset():
    if not os.path.exists(DATA_DIR):
        with st.spinner("⬇️  Mengunduh dataset MovieLens..."):
            urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(".")
            os.remove(ZIP_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    ratings_df = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    movies_df  = pd.read_csv(f"{DATA_DIR}/movies.csv")
    return ratings_df, movies_df


@st.cache_data(show_spinner=False)
def get_filtered_df(ratings_df):
    user_counts  = ratings_df.groupby("userId").size()
    active_users = user_counts[user_counts >= MIN_RATINGS].index
    return ratings_df[ratings_df["userId"].isin(active_users)].copy()


@st.cache_resource(show_spinner=False)
def load_or_train_model(filtered_df):
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split

    if Path(MODEL_PATH).exists():
        return joblib.load(MODEL_PATH)

    with st.spinner("🤖 Melatih model SVD (pertama kali, mohon tunggu ~1–2 menit)..."):
        reader   = Reader(rating_scale=(0.5, 5.0))
        data     = Dataset.load_from_df(filtered_df[["userId", "movieId", "rating"]], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

        model = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
        model.fit(trainset)
        joblib.dump(model, MODEL_PATH)
        return model


# ─── Recommendation logic ────────────────────────────────────────────────────
def get_recommendations(user_id, model, filtered_df, movies_df, n=10):
    rated_movies = set(filtered_df[filtered_df["userId"] == user_id]["movieId"])
    all_movies   = set(movies_df["movieId"])
    unrated      = all_movies - rated_movies

    if not rated_movies:
        return None, 0

    preds = [(mid, model.predict(user_id, mid).est) for mid in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_n = preds[:n]

    results = []
    for rank, (mid, est) in enumerate(top_n, 1):
        row = movies_df[movies_df["movieId"] == mid]
        if len(row) > 0:
            results.append({
                "rank"  : rank,
                "title" : row["title"].values[0],
                "genres": row["genres"].values[0].replace("|", " · "),
                "score" : round(est, 2),
                "stars" : "⭐" * round(est),
            })
    return results, len(rated_movies)


def get_user_history(user_id, filtered_df, movies_df, n=10):
    user_ratings = filtered_df[filtered_df["userId"] == user_id].copy()
    user_ratings = user_ratings.sort_values("rating", ascending=False).head(n)
    merged = user_ratings.merge(movies_df, on="movieId")
    merged["genres"] = merged["genres"].str.replace("|", " · ", regex=False)
    return merged[["title", "genres", "rating"]]


# ─── EDA helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_eda(ratings_df, movies_df):
    n_users   = ratings_df["userId"].nunique()
    n_movies  = ratings_df["movieId"].nunique()
    n_ratings = len(ratings_df)
    sparsity  = 1 - (n_ratings / (n_users * n_movies))
    avg_rating = ratings_df["rating"].mean()
    return n_users, n_movies, n_ratings, sparsity, avg_rating


def plot_rating_distribution(ratings_df):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    rc = ratings_df["rating"].value_counts().sort_index()
    bars = ax.bar(rc.index.astype(str), rc.values, color="#e94560", edgecolor="#0f1117", width=0.65)
    ax.set_title("Distribusi Rating", color="#e2e8f0", fontsize=12, pad=10)
    ax.set_xlabel("Nilai Rating", color="#718096")
    ax.set_ylabel("Jumlah Rating", color="#718096")
    ax.tick_params(colors="#718096")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{int(bar.get_height()):,}", ha="center", va="bottom",
                fontsize=7, color="#a0aec0")
    plt.tight_layout()
    return fig


def plot_top_movies(ratings_df, movies_df):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    top = (ratings_df.groupby("movieId").size()
           .reset_index(name="count")
           .merge(movies_df[["movieId", "title"]], on="movieId")
           .nlargest(10, "count"))
    top["short"] = top["title"].str[:28]
    ax.barh(top["short"][::-1], top["count"][::-1], color="#4a6cf7", edgecolor="#0f1117")
    ax.set_title("Top 10 Film Paling Banyak Dirating", color="#e2e8f0", fontsize=12, pad=10)
    ax.set_xlabel("Jumlah Rating", color="#718096")
    ax.tick_params(colors="#718096")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    plt.tight_layout()
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("<p style='color:#718096;font-size:0.85rem;'>Film Recommendation System — SVD</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Rekomendasi", "📊 Eksplorasi Data", "ℹ️ Tentang Model"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<p style='color:#4a5568;font-size:0.78rem;'>Capstone · Pijak × IBM SkillsBuild<br>Dataset: MovieLens Latest Small</p>", unsafe_allow_html=True)


# ─── Init data ───────────────────────────────────────────────────────────────
download_dataset()
ratings_df, movies_df = load_data()
filtered_df = get_filtered_df(ratings_df)
valid_users = sorted(filtered_df["userId"].unique())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""
    <div class="hero-banner">
        <h1>🎬 CineMatch</h1>
        <p>Sistem Rekomendasi Film berbasis <strong>Singular Value Decomposition (SVD)</strong></p>
    </div>
    """, unsafe_allow_html=True)

    n_users, n_movies, n_ratings, sparsity, avg_rating = compute_eda(ratings_df, movies_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="value">{n_ratings:,}</div><div class="label">Total Rating</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="value">{n_users:,}</div><div class="label">Pengguna</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="value">{n_movies:,}</div><div class="label">Film</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="value">{avg_rating:.2f}⭐</div><div class="label">Rata-rata Rating</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cara Kerja CineMatch</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#e94560">① Matriks User-Item</strong><br><br>
        Setiap pengguna dan film direpresentasikan dalam matriks rating yang sangat sparse (~98.3% kosong).
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#e94560">② Dekomposisi SVD</strong><br><br>
        SVD memecah matriks menjadi faktor laten (R ≈ U·Σ·Vᵀ), mengungkap pola tersembunyi selera pengguna.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#e94560">③ Rekomendasi Personal</strong><br><br>
        Model memprediksi rating film yang belum ditonton dan merekomendasikan 10 terbaik untuk setiap pengguna.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Start</div>', unsafe_allow_html=True)
    st.info("👈 Buka halaman **🔍 Rekomendasi** di sidebar, masukkan User ID, dan dapatkan 10 rekomendasi film personal!")

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_rating_distribution(ratings_df), use_container_width=True)
    with col_b:
        st.pyplot(plot_top_movies(ratings_df, movies_df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REKOMENDASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Rekomendasi":
    st.markdown('<div class="section-title">🔍 Dapatkan Rekomendasi Film</div>', unsafe_allow_html=True)

    with st.spinner("🤖 Memuat / melatih model SVD..."):
        model = load_or_train_model(filtered_df)

    col_input, col_info = st.columns([1, 2])
    with col_input:
        user_id = st.number_input(
            "User ID",
            min_value=int(min(valid_users)),
            max_value=int(max(valid_users)),
            value=int(valid_users[0]),
            step=1,
            help=f"Pilih User ID ({min(valid_users)}–{max(valid_users)}). Contoh: 1, 5, 42, 100, 300",
        )
        n_recs = st.slider("Jumlah Rekomendasi", min_value=5, max_value=20, value=10)
        run_btn = st.button("🎬 Tampilkan Rekomendasi", use_container_width=True, type="primary")

    with col_info:
        st.markdown("""
        <div class="info-box">
        <strong>ℹ️ Petunjuk</strong><br><br>
        • Masukkan <b>User ID</b> yang valid dari dataset MovieLens.<br>
        • Klik <b>Tampilkan Rekomendasi</b> untuk melihat film yang diprediksi sesuai selera user.<br>
        • Rekomendasi dihasilkan dari film-film yang <i>belum</i> pernah dirating oleh user tersebut.
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        if user_id not in valid_users:
            st.error(f"⚠️ User ID **{user_id}** tidak ditemukan. Pastikan menggunakan user yang aktif (≥ {MIN_RATINGS} rating).")
        else:
            with st.spinner(f"⚙️ Menghitung rekomendasi untuk User #{user_id}..."):
                recs, n_rated = get_recommendations(user_id, model, filtered_df, movies_df, n=n_recs)

            if not recs:
                st.warning("Tidak ada rekomendasi yang bisa dihasilkan untuk user ini.")
            else:
                st.success(f"✅ Menampilkan {len(recs)} rekomendasi berdasarkan **{n_rated} film** yang telah dirating oleh User #{user_id}.")

                tab_recs, tab_history = st.tabs(["🎬 Rekomendasi", "📋 Riwayat Rating User"])

                with tab_recs:
                    for r in recs:
                        stars_full  = "⭐" * int(r["score"])
                        stars_empty = "☆" * (5 - int(r["score"]))
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-rank">#{r['rank']}</div>
                            <div style="flex:1">
                                <div class="rec-title">{r['title']}</div>
                                <div class="rec-genre">{r['genres']}</div>
                            </div>
                            <div class="rec-rating">
                                <div class="rec-stars">{stars_full}{stars_empty}</div>
                                <div class="rec-score">Prediksi: {r['score']}/5.0</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Download button
                    df_export = pd.DataFrame(recs).rename(columns={
                        "rank": "Rank", "title": "Judul Film",
                        "genres": "Genre", "score": "Prediksi Rating", "stars": "Bintang"
                    })
                    csv = df_export.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Unduh sebagai CSV", csv,
                                       file_name=f"rekomendasi_user_{user_id}.csv",
                                       mime="text/csv")

                with tab_history:
                    history = get_user_history(user_id, filtered_df, movies_df, n=15)
                    if history.empty:
                        st.info("Tidak ada riwayat rating.")
                    else:
                        st.dataframe(
                            history.rename(columns={"title": "Judul Film", "genres": "Genre", "rating": "Rating"}),
                            use_container_width=True, hide_index=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EKSPLORASI DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Eksplorasi Data":
    st.markdown('<div class="section-title">📊 Eksplorasi Data (EDA)</div>', unsafe_allow_html=True)

    n_users, n_movies, n_ratings, sparsity, avg_rating = compute_eda(ratings_df, movies_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sparsity Matriks", f"{sparsity:.2%}")
    col2.metric("Avg Rating", f"{avg_rating:.3f} / 5.0")
    col3.metric("User Aktif (≥20 rating)", f"{len(valid_users):,}")

    tab1, tab2, tab3 = st.tabs(["Distribusi Rating", "Distribusi per User/Film", "Top Film"])

    with tab1:
        st.pyplot(plot_rating_distribution(ratings_df), use_container_width=True)
        st.markdown("""
        <div class="info-box">
        Rating didominasi oleh nilai <b>4.0</b> dan <b>3.0</b>. Pengguna cenderung memberi rating positif (sebagian besar di atas 3.0), 
        yang merupakan pola umum dalam dataset rating film.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        ratings_per_user  = ratings_df.groupby("userId").size()
        ratings_per_movie = ratings_df.groupby("movieId").size()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="#718096")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a4a")

        ax1.hist(ratings_per_user, bins=40, color="#4a6cf7", edgecolor="#0f1117")
        ax1.axvline(ratings_per_user.median(), color="#e94560", linestyle="--",
                    label=f"Median: {ratings_per_user.median():.0f}")
        ax1.set_title("Rating per User", color="#e2e8f0")
        ax1.set_xlabel("Jumlah Rating", color="#718096")
        ax1.set_ylabel("Jumlah User", color="#718096")
        ax1.legend(labelcolor="#a0aec0", facecolor="#1a1a2e", edgecolor="#2a2a4a")

        ax2.hist(ratings_per_movie, bins=50, color="#38b2ac", edgecolor="#0f1117")
        ax2.axvline(ratings_per_movie.median(), color="#e94560", linestyle="--",
                    label=f"Median: {ratings_per_movie.median():.0f}")
        ax2.set_title("Rating per Film", color="#e2e8f0")
        ax2.set_xlabel("Jumlah Rating Diterima", color="#718096")
        ax2.set_ylabel("Jumlah Film", color="#718096")
        ax2.legend(labelcolor="#a0aec0", facecolor="#1a1a2e", edgecolor="#2a2a4a")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab3:
        st.pyplot(plot_top_movies(ratings_df, movies_df), use_container_width=True)

        top15 = (ratings_df.groupby("movieId").size()
                 .reset_index(name="Jumlah Rating")
                 .merge(movies_df[["movieId", "title", "genres"]], on="movieId")
                 .nlargest(15, "Jumlah Rating")
                 .rename(columns={"title": "Judul Film", "genres": "Genre"})
                 [["Judul Film", "Genre", "Jumlah Rating"]])
        top15["Genre"] = top15["Genre"].str.replace("|", " · ", regex=False)
        st.dataframe(top15, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TENTANG MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Model":
    st.markdown('<div class="section-title">ℹ️ Tentang Model CineMatch</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#e94560">Algoritma: SVD (Singular Value Decomposition)</strong><br><br>
        SVD mendekomposisi matriks user-item <b>R</b> menjadi:<br><br>
        <code>R ≈ U · Σ · Vᵀ</code><br><br>
        <ul>
          <li><b>U</b> — matriks preferensi user (610 × k)</li>
          <li><b>Σ</b> — matriks nilai singular</li>
          <li><b>Vᵀ</b> — matriks karakteristik film (k × 9.742)</li>
          <li><b>k</b> — jumlah latent factors (default: 50)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-box">
        <strong style="color:#e94560">Hyperparameter Terbaik (Grid Search)</strong><br><br>
        <table style="width:100%;color:#a0aec0">
          <tr><td>n_factors</td><td><b style="color:#e2e8f0">50</b></td></tr>
          <tr><td>n_epochs</td><td><b style="color:#e2e8f0">30</b></td></tr>
          <tr><td>lr_all</td><td><b style="color:#e2e8f0">0.005</b></td></tr>
          <tr><td>reg_all</td><td><b style="color:#e2e8f0">0.02</b></td></tr>
          <tr><td>random_state</td><td><b style="color:#e2e8f0">42</b></td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Metrik Evaluasi Target</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="value">< 1.0</div>
          <div class="label">Target RMSE<br><small>Root Mean Squared Error</small></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="value">> 60%</div>
          <div class="label">Target Precision@10<br><small>Proporsi film relevan dari Top-10</small></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>MovieLens Latest Small</b> — GroupLens Research, University of Minnesota<br><br>
    • 100,836 ratings dari 610 pengguna terhadap 9,742 film<br>
    • Skala rating: 0.5 – 5.0 (increment 0.5)<br>
    • Filter aktif: hanya user dengan ≥ 20 rating<br>
    • Sumber: <a href="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip" target="_blank" style="color:#4a6cf7">files.grouplens.org</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Alur Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣ Download & Load", "Dataset MovieLens diunduh otomatis dan dimuat ke DataFrame."),
        ("2️⃣ EDA", "Eksplorasi distribusi rating, sparsity, dan karakteristik dataset."),
        ("3️⃣ Preprocessing", "Filter user aktif (≥ 20 rating), lalu train-test split 80/20."),
        ("4️⃣ Training SVD", "Model SVD dilatih dengan hyperparameter optimal dari Grid Search."),
        ("5️⃣ Evaluasi", "RMSE dan Precision@10 dihitung pada test set."),
        ("6️⃣ Rekomendasi", "Model memprediksi rating film yang belum ditonton, lalu sort Top-N."),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div class="info-box">
        <strong style="color:#e94560">{title}</strong><br>{desc}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='color:#4a5568;text-align:center;font-size:0.85rem;'>CineMatch Capstone Project · Pijak × IBM SkillsBuild · 2025</p>", unsafe_allow_html=True)
