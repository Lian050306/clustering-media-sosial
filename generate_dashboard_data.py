# generate_dashboard_data.py (VERSI YANG SUDAH DIPERBAIKI)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Buat folder static jika belum ada
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('templates'):
    os.makedirs('templates')

print("="*60)
print("MEMBANGUN DASHBOARD DARI HASIL ANALISIS COLAB")
print("="*60)

# 1. BACA DATA
print("\n📂 Membaca data...")
df = pd.read_csv('data/dataset.csv', sep=';')
print(f"   Total responden: {len(df)}")

# 2. FUNGSI KONVERSI JAM
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

# 3. KONVERSI DATA
print("\n🔄 Mengkonversi data...")
df['jam_medsos'] = df['Berapa lama Anda menggunakan Media Sosial dalam sehari?'].apply(konversi_jam)
df['jam_tidur'] = df['Berapa lama Anda tidur dalam sehari?'].apply(konversi_jam)

# 4. ENCODING
le_ipk = LabelEncoder()
le_pengaruh = LabelEncoder()
le_tidur = LabelEncoder()

df['ipk_encoded'] = le_ipk.fit_transform(df['Apakah Anda merasa prestasi akademik (IPK) Anda baik?'].astype(str))
df['pengaruh_encoded'] = le_pengaruh.fit_transform(df['Apakah penggunaan Media Sosial dapat mempengaruhi prestasi akademik Anda?'].astype(str))
df['tidur_encoded'] = le_tidur.fit_transform(df['Apakah penggunaan Media Sosial berpengaruh terhadap jam tidur Anda?'].astype(str))

# 5. CLUSTERING
print("\n🤖 Melakukan clustering...")
fitur = ['jam_medsos', 'jam_tidur', 'ipk_encoded', 'pengaruh_encoded', 'tidur_encoded']
X = df[fitur]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gunakan 10 klaster
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df['cluster'])
print(f"   Silhouette Score: {sil_score:.4f}")

# Simpan model
os.makedirs('models', exist_ok=True)
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# ============================================================
# GRAFIK 1: ELBOW METHOD
# ============================================================
print("\n📊 Membuat grafik Elbow Method...")
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Jumlah Klaster (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method untuk K Optimal', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Jumlah Klaster (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score untuk K Optimal', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ static/elbow_method.png")

# ============================================================
# GRAFIK 2: DISTRIBUSI KLASTER
# ============================================================
print("\n📊 Membuat grafik distribusi klaster...")
plt.figure(figsize=(12, 6))
cluster_counts = df['cluster'].value_counts().sort_index()
colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Jumlah Responden', fontsize=12)
plt.title('Distribusi Jumlah Responden per Klaster', fontsize=14, fontweight='bold')
plt.xticks(range(len(cluster_counts)), [f'Klaster {i}' for i in cluster_counts.index])
for bar, count in zip(bars, cluster_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('static/cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ static/cluster_distribution.png")

# ============================================================
# GRAFIK 3: SCATTER PLOT
# ============================================================
print("\n📊 Membuat grafik scatter plot...")
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['jam_medsos'], df['jam_tidur'], 
                      c=df['cluster'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
plt.xlabel('Penggunaan Media Sosial (jam/hari)', fontsize=12)
plt.ylabel('Jam Tidur (jam/hari)', fontsize=12)
plt.title('Klastering Berdasarkan Penggunaan Medsos vs Jam Tidur', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, label='Cluster')
cbar.set_ticks(range(optimal_k))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/scatter_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ static/scatter_plot.png")

# ============================================================
# GRAFIK 4: BOXPLOT
# ============================================================
print("\n📊 Membuat grafik boxplot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df.boxplot(column='jam_medsos', by='cluster', ax=axes[0])
axes[0].set_title('Penggunaan Media Sosial per Klaster', fontsize=12)
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Jam per Hari')

df.boxplot(column='jam_tidur', by='cluster', ax=axes[1])
axes[1].set_title('Jam Tidur per Klaster', fontsize=12)
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Jam per Hari')

plt.suptitle('Distribusi Data per Klaster', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ static/boxplots.png")

# ============================================================
# GRAFIK 5: PIE CHARTS (DIPERBAIKI)
# ============================================================
print("\n📊 Membuat grafik pie charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart pengaruh medsos ke tidur
pengaruh_tidur = df['Apakah penggunaan Media Sosial berpengaruh terhadap jam tidur Anda?'].value_counts()
colors_pie = ['#e74c3c', '#2ecc71']

# Hapus explode jika tidak sesuai
axes[0].pie(pengaruh_tidur.values, labels=pengaruh_tidur.index, autopct='%1.1f%%', 
            colors=colors_pie[:len(pengaruh_tidur)], startangle=90)
axes[0].set_title('Apakah Medsos Berpengaruh ke Jam Tidur?', fontsize=12, fontweight='bold')

# Pie chart pengaruh medsos ke prestasi
pengaruh_prestasi = df['Apakah penggunaan Media Sosial dapat mempengaruhi prestasi akademik Anda?'].value_counts()
axes[1].pie(pengaruh_prestasi.values, labels=pengaruh_prestasi.index, autopct='%1.1f%%', 
            colors=colors_pie[:len(pengaruh_prestasi)], startangle=90)
axes[1].set_title('Apakah Medsos Berpengaruh ke Prestasi?', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('static/pie_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ static/pie_charts.png")

# ============================================================
# GRAFIK 6: HISTOGRAM
# ============================================================
print("\n📊 Membuat grafik histogram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['jam_medsos'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Jam per Hari', fontsize=12)
axes[0].set_ylabel('Frekuensi', fontsize=12)
axes[0].set_title('Distribusi Penggunaan Media Sosial', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['jam_tidur'], bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Jam per Hari', fontsize=12)
axes[1].set_ylabel('Frekuensi', fontsize=12)
axes[1].set_title('Distribusi Jam Tidur', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ static/histograms.png")

# ============================================================
# TABEL KARAKTERISTIK KLASTER
# ============================================================
print("\n📊 Membuat tabel karakteristik klaster...")
cluster_summary = df.groupby('cluster').agg({
    'jam_medsos': ['mean', 'min', 'max'],
    'jam_tidur': ['mean', 'min', 'max'],
    'ipk_encoded': 'mean',
    'pengaruh_encoded': 'mean',
    'tidur_encoded': 'mean'
}).round(2)

cluster_summary.columns = ['Rata2 Medsos', 'Min Medsos', 'Max Medsos',
                           'Rata2 Tidur', 'Min Tidur', 'Max Tidur',
                           'Proporsi IPK Baik', 'Proporsi Pengaruh ke Prestasi', 'Proporsi Pengaruh ke Tidur']

# Interpretasi
interpretasi = []
for i in range(optimal_k):
    medsos = cluster_summary.iloc[i]['Rata2 Medsos']
    tidur = cluster_summary.iloc[i]['Rata2 Tidur']
    
    if medsos > 6:
        tipe = "Pengguna Berat"
    elif medsos > 3:
        tipe = "Pengguna Sedang"
    else:
        tipe = "Pengguna Ringan"
    
    if tidur < 6:
        kualitas = "Kurang Tidur"
    elif tidur < 8:
        kualitas = "Tidur Cukup"
    else:
        kualitas = "Tidur Ideal"
    
    interpretasi.append(f"{tipe} - {kualitas}")

cluster_summary['Tipe'] = interpretasi
cluster_summary['Jumlah'] = df['cluster'].value_counts().sort_index().values
cluster_summary.index.name = 'cluster'
cluster_summary = cluster_summary.reset_index()

# Simpan ke CSV
cluster_summary.to_csv('static/cluster_summary.csv', index=False)
print("   ✅ static/cluster_summary.csv")

# ============================================================
# SIMPAN STATISTIK UMUM
# ============================================================
print("\n📊 Menyimpan statistik umum...")
stats = {
    'total_responden': len(df),
    'rata2_medsos': round(df['jam_medsos'].mean(), 2),
    'rata2_tidur': round(df['jam_tidur'].mean(), 2),
    'silhouette_score': round(sil_score, 4),
    'optimal_k': optimal_k,
    'cluster_counts': {int(k): int(v) for k, v in df['cluster'].value_counts().to_dict().items()},
    'persen_pengaruh_tidur': round((df['tidur_encoded'].sum() / len(df) * 100), 1),
    'persen_pengaruh_prestasi': round((df['pengaruh_encoded'].sum() / len(df) * 100), 1)
}
with open('static/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("   ✅ static/stats.json")

print("\n" + "="*60)
print("✅ SEMUA GRAFIK DAN DATA BERHASIL DISIMPAN!")
print("="*60)
print("\n📁 File tersimpan di folder 'static/':")
for f in os.listdir('static'):
    print(f"   - {f}")