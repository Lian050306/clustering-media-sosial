# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Media Sosial & Jam Tidur",
    page_icon="📊",
    layout="wide"
)

# Judul
st.title("📊 Analisis Penggunaan Media Sosial dan Jam Tidur")
st.markdown("### Hasil Clustering dengan Metode K-Means")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv', sep=';')
    return df

# Fungsi konversi jam
def konversi_jam(teks):
    teks = str(teks).lower()
    if 'lebih dari 6' in teks:
        return 8
    elif 'kurang dari 6' in teks:
        return 4
    elif 'lebih dari 2' in teks:
        return 3
    elif 'kurang dari 8' in teks:
        return 6
    else:
        return 5

# Load dan proses data
df = load_data()

# Konversi data
df['jam_medsos'] = df['Berapa lama Anda menggunakan Media Sosial dalam sehari?'].apply(konversi_jam)
df['jam_tidur'] = df['Berapa lama Anda tidur dalam sehari?'].apply(konversi_jam)

# Encoding
le_ipk = LabelEncoder()
le_pengaruh = LabelEncoder()
le_tidur = LabelEncoder()

df['ipk_encoded'] = le_ipk.fit_transform(df['Apakah Anda merasa prestasi akademik (IPK) Anda baik?'].astype(str))
df['pengaruh_encoded'] = le_pengaruh.fit_transform(df['Apakah penggunaan Media Sosial dapat mempengaruhi prestasi akademik Anda?'].astype(str))
df['tidur_encoded'] = le_tidur.fit_transform(df['Apakah penggunaan Media Sosial berpengaruh terhadap jam tidur Anda?'].astype(str))

# Clustering
fitur = ['jam_medsos', 'jam_tidur', 'ipk_encoded', 'pengaruh_encoded', 'tidur_encoded']
X = df[fitur]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gunakan 10 klaster (atau sesuaikan)
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

df['cluster'] = kmeans.fit_predict(X_scaled)

# ========== HITUNG PROPORSISI YANG BENAR UNTUK TABEL ==========
cluster_stats = df.groupby('cluster').agg({
    'jam_medsos': 'mean',
    'jam_tidur': 'mean',
    'ipk_encoded': lambda x: (x.sum() / len(x)) * 100,
    'pengaruh_encoded': lambda x: (x.sum() / len(x)) * 100,
    'tidur_encoded': lambda x: (x.sum() / len(x)) * 100
}).round(1)

cluster_stats.columns = ['Rata2 Medsos', 'Rata2 Tidur', 
                         'IPK Baik (%)', 'Pengaruh ke Prestasi (%)', 
                         'Pengaruh ke Tidur (%)']
cluster_stats['Jumlah'] = df['cluster'].value_counts().sort_index()

# Silhouette score
sil_score = silhouette_score(X_scaled, df['cluster'])

# ============================================================
# SIDEBAR - STATISTIK
# ============================================================
st.sidebar.title("📊 Statistik Dataset")
st.sidebar.metric("Total Responden", len(df))
st.sidebar.metric("Rata-rata Penggunaan Medsos", f"{df['jam_medsos'].mean():.1f} jam/hari")
st.sidebar.metric("Rata-rata Jam Tidur", f"{df['jam_tidur'].mean():.1f} jam/hari")
st.sidebar.metric("Silhouette Score", f"{sil_score:.4f}")
st.sidebar.metric("Jumlah Klaster Optimal", optimal_k)
st.sidebar.markdown("---")
st.sidebar.info("🔬 Metode: K-Means Clustering")

# ============================================================
# ROW 1: GRAFIK ELBOW & SILHOUETTE
# ============================================================
st.subheader("📈 Penentuan Jumlah Klaster Optimal")

col1, col2 = st.columns(2)

with col1:
    # Elbow Method
    inertia = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_range, inertia, 'bo-', linewidth=2)
    ax.set_xlabel('Jumlah Klaster (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    # Silhouette Score
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_range, sil_scores, 'ro-', linewidth=2)
    ax.set_xlabel('Jumlah Klaster (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")

# ============================================================
# ROW 2: DISTRIBUSI KLASTER & SCATTER PLOT
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribusi Responden per Klaster")
    cluster_counts = df['cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Jumlah Responden')
    ax.set_title('Distribusi 10 Klaster')
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count), ha='center')
    st.pyplot(fig)

with col2:
    st.subheader("🔄 Hubungan Medsos vs Jam Tidur")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df['jam_medsos'], df['jam_tidur'], 
                         c=df['cluster'], cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Penggunaan Media Sosial (jam/hari)')
    ax.set_ylabel('Jam Tidur (jam/hari)')
    ax.set_title('Scatter Plot per Klaster')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)

st.markdown("---")

# ============================================================
# ROW 3: BOXPLOT
# ============================================================
st.subheader("📦 Distribusi Data per Klaster")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column='jam_medsos', by='cluster', ax=ax)
    ax.set_title('Penggunaan Media Sosial per Klaster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Jam per Hari')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column='jam_tidur', by='cluster', ax=ax)
    ax.set_title('Jam Tidur per Klaster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Jam per Hari')
    st.pyplot(fig)

st.markdown("---")

# ============================================================
# ROW 4: PIE CHARTS
# ============================================================
st.subheader("🥧 Persepsi Responden")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    pengaruh_tidur = df['Apakah penggunaan Media Sosial berpengaruh terhadap jam tidur Anda?'].value_counts()
    colors_pie = ['#e74c3c', '#2ecc71']
    ax.pie(pengaruh_tidur.values, labels=pengaruh_tidur.index, autopct='%1.1f%%', 
           colors=colors_pie[:len(pengaruh_tidur)], startangle=90)
    ax.set_title('Apakah Medsos Berpengaruh ke Jam Tidur?')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 5))
    pengaruh_prestasi = df['Apakah penggunaan Media Sosial dapat mempengaruhi prestasi akademik Anda?'].value_counts()
    ax.pie(pengaruh_prestasi.values, labels=pengaruh_prestasi.index, autopct='%1.1f%%', 
           colors=colors_pie[:len(pengaruh_prestasi)], startangle=90)
    ax.set_title('Apakah Medsos Berpengaruh ke Prestasi?')
    st.pyplot(fig)

st.markdown("---")

# ============================================================
# ROW 5: TABEL KARAKTERISTIK KLASTER
# ============================================================
st.subheader("📋 Karakteristik Setiap Klaster (10 Klaster)")

# KODE BARU (INI YANG DIPAKE)
cluster_stats = df.groupby('cluster').agg({
    'jam_medsos': 'mean',
    'jam_tidur': 'mean',
    'ipk_encoded': lambda x: (x.sum() / len(x)) * 100,
    'pengaruh_encoded': lambda x: (x.sum() / len(x)) * 100,
    'tidur_encoded': lambda x: (x.sum() / len(x)) * 100
}).round(1)

cluster_stats.columns = ['Rata2 Medsos', 'Rata2 Tidur', 
                         'IPK Baik (%)', 'Pengaruh ke Prestasi (%)', 
                         'Pengaruh ke Tidur (%)']
cluster_stats['Jumlah'] = df['cluster'].value_counts().sort_index()

st.dataframe(cluster_stats, use_container_width=True)

cluster_stats.columns = ['Rata2 Medsos', 'Rata2 Tidur', 'Proporsi IPK Baik', 
                           'Proporsi Pengaruh ke Prestasi', 'Proporsi Pengaruh ke Tidur']
cluster_stats['Jumlah'] = df['cluster'].value_counts().sort_index()

# Interpretasi
tipe_klaster = []
for i in range(optimal_k):
    medsos = cluster_stats.iloc[i]['Rata2 Medsos']
    tidur = cluster_stats.iloc[i]['Rata2 Tidur']
    
    if medsos > 6:
        tipe = "🔴 Pengguna Berat"
    elif medsos > 3:
        tipe = "🟡 Pengguna Sedang"
    else:
        tipe = "🟢 Pengguna Ringan"
    
    if tidur < 6:
        kualitas = "Kurang Tidur"
    elif tidur < 8:
        kualitas = "Tidur Cukup"
    else:
        kualitas = "Tidur Ideal"
    
    tipe_klaster.append(f"{tipe} - {kualitas}")

cluster_stats['Tipe'] = tipe_klaster

# Tampilkan tabel
st.dataframe(cluster_stats, use_container_width=True)

# ============================================================
# ROW 6: HISTOGRAM
# ============================================================
st.subheader("📊 Distribusi Data")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['jam_medsos'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Jam per Hari')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Distribusi Penggunaan Media Sosial')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['jam_tidur'], bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Jam per Hari')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Distribusi Jam Tidur')
    st.pyplot(fig)

st.markdown("---")
st.caption("🔬 Analisis menggunakan K-Means Clustering | Data dari 544 responden")