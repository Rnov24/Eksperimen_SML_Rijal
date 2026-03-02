# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: smsml
#     language: python
#     name: python3
# ---

# %% [markdown] id="kZLRMFl0JyyQ"
# # **1. Perkenalan Dataset**
#
# ## Sleep Efficiency Dataset
#
# **Sumber:** Kaggle
# **Link:** [https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency](https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency)
#
# ### Deskripsi Dataset
# Dataset ini berisi informasi tentang sekelompok subjek uji dan pola tidur mereka. Setiap subjek diidentifikasi dengan "Subject ID" unik, serta dicatat usia dan jenis kelaminnya. Fitur "Bedtime" dan "Wakeup time" menunjukkan waktu tidur dan bangun setiap hari, sementara "Sleep duration" mencatat total durasi tidur dalam jam. Fitur "Sleep efficiency" mengukur proporsi waktu di tempat tidur yang benar-benar dihabiskan untuk tidur.
#
# Fitur "REM sleep percentage", "Deep sleep percentage", dan "Light sleep percentage" menunjukkan waktu yang dihabiskan di setiap tahap tidur. "Awakenings" mencatat berapa kali subjek terbangun di malam hari. Dataset juga mencakup konsumsi kafein dan alkohol 24 jam sebelum tidur, status merokok, dan frekuensi olahraga.
#
# ### Sumber Data
# Dataset dikumpulkan sebagai bagian dari studi yang dilakukan di Maroko oleh sekelompok mahasiswa teknik kecerdasan buatan dari ENSIAS (Ecole Nationale Supérieure d'Informatique et d'Analyse des Systèmes).
#
# ### Metodologi Pengumpulan
# Studi ini bertujuan untuk menyelidiki dampak faktor gaya hidup seperti kafein, alkohol, dan olahraga terhadap pola dan kualitas tidur. Data dikumpulkan menggunakan kombinasi survei self-report, aktigrafi, dan polisomnografi (teknik monitoring tidur) selama beberapa bulan.
#
# ### Informasi Dasar
# - **Jumlah sampel:** 452
# - **Jumlah kolom:** 15
# - **Target variabel:** Sleep efficiency (proporsi 0-1)
#

# %% [markdown] id="fKADPWcFKlj3"
# # **2. Import Library**

# %% [markdown] id="LgA3ERnVn84N"
# Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning.

# %% id="BlmvjLY9M4Yj"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder


# %% [markdown] id="f3YIEnAFKrKL"
# # **3. Memuat Dataset**

# %% [markdown] id="Ey3ItwTen_7E"
# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
#
# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.
#
# Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan

# %% id="GHCGNTyrM5fS"
# muat dataset dari file CSV
df = pd.read_csv('sleep_efficiency_raw.csv')
print(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
df.head()

# %% id="GHCGNTyrM5fS_2"
# cek struktur data dan tipe kolomnya
df.info()

# %% id="GHCGNTyrM5fS_3"
# cek beberapa baris terakhir juga
df.tail()

# %% [markdown] id="bgZkbJLpK9UR"
# # **4. Exploratory Data Analysis (EDA)**
#
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.
#
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# %% id="dKeejtvxM6X1"
# statistik deskriptif buat dapet gambaran umum data
df.describe()

# %% id="eda_missing"
# cek missing values per kolom
missing = df.isnull().sum()
print("Missing values per kolom:")
print(missing[missing > 0])
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# %% id="eda_duplikat"
# cek duplikat
print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

# %% id="eda_distribusi_target"
# distribusi target (Sleep efficiency)
plt.figure(figsize=(10, 5))
sns.histplot(df['Sleep efficiency'], bins=20, kde=True, color='steelblue')
plt.title('Distribusi Sleep Efficiency')
plt.xlabel('Sleep Efficiency')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.show()

# %% id="eda_korelasi"
# heatmap korelasi antar fitur numerik
plt.figure(figsize=(12, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# %% id="eda_distribusi_age"
# distribusi umur responden
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=15, kde=True, color='coral')
plt.title('Distribusi Umur')
plt.xlabel('Age')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.show()

# %% id="eda_gender"
# perbandingan gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Jumlah Data per Gender')
plt.tight_layout()
plt.show()

# %% id="eda_smoking"
# pengaruh smoking terhadap sleep efficiency
plt.figure(figsize=(8, 5))
sns.boxplot(x='Smoking status', y='Sleep efficiency', data=df, palette='Set3')
plt.title('Sleep Efficiency berdasarkan Smoking Status')
plt.tight_layout()
plt.show()

# %% id="eda_sleep_duration"
# scatter plot sleep duration vs sleep efficiency
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Sleep duration', y='Sleep efficiency', hue='Gender', data=df, alpha=0.6)
plt.title('Sleep Duration vs Sleep Efficiency')
plt.tight_layout()
plt.show()

# %% [markdown] id="cpgHfgnSK3ip"
# # **5. Data Preprocessing**

# %% [markdown] id="COf8KUPXLg5r"
# Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.
#
# Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.
#
# Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:
# 1. Menghapus atau Menangani Data Kosong (Missing Values)
# 2. Menghapus Data Duplikat
# 3. Normalisasi atau Standarisasi Fitur
# 4. Deteksi dan Penanganan Outlier
# 5. Encoding Data Kategorikal
# 6. Binning (Pengelompokan Data)
#
# Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur.

# %% id="Og8pGV0-iDLz"
# ===== PERSIAPAN =====
# buat copy dataframe biar data asli tetep aman
df_clean = df.copy()

# buang kolom yang nggak relevan buat modelling
# ID cuma identifier, Bedtime dan Wakeup time formatnya datetime string, ribet dan nggak terlalu berguna
cols_to_drop = ['ID', 'Bedtime', 'Wakeup time']
df_clean = df_clean.drop(columns=cols_to_drop)

print(f"Kolom yang dibuang: {cols_to_drop}")
print(f"Sisa kolom ({len(df_clean.columns)}): {list(df_clean.columns)}")
print(f"Shape: {df_clean.shape}")

# %% id="prep_step1_missing"
# ===== STEP 1: Menangani Data Kosong (Missing Values) =====

# lihat kondisi missing values sebelum ditangani
missing_before = df_clean.isnull().sum()
missing_cols = missing_before[missing_before > 0]

print("Kondisi missing values saat ini:")
print("-" * 45)
for col, count in missing_cols.items():
    pct = (count / len(df_clean)) * 100
    print(f"  {col:30s} -> {count:3d} missing ({pct:.1f}%)")
print(f"\nTotal sel kosong: {df_clean.isnull().sum().sum()}")

# visualisasi missing values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# bar chart missing values
missing_cols.plot(kind='bar', ax=axes[0], color='salmon', edgecolor='black')
axes[0].set_title('Jumlah Missing Values per Kolom')
axes[0].set_ylabel('Jumlah')
axes[0].tick_params(axis='x', rotation=45)

# heatmap missing values
sns.heatmap(df_clean.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Pola Missing Values (kuning = missing)')

plt.tight_layout()
plt.show()

# %% id="prep_step1_fill"
# isi missing values
# strategi: pake median buat numerik (lebih robust terhadap outlier daripada mean)
for col in df_clean.columns:
    null_count = df_clean[col].isnull().sum()
    if null_count > 0:
        if df_clean[col].dtype in ['float64', 'int64']:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"[MEDIAN] '{col}' -> {null_count} missing diisi dengan {median_val}")
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"[MODE]   '{col}' -> {null_count} missing diisi dengan '{mode_val}'")

print(f"\nSisa missing values: {df_clean.isnull().sum().sum()} (harusnya 0)")

# %% id="prep_step2_duplikat"
# ===== STEP 2: Menghapus Data Duplikat =====

n_before = len(df_clean)
duplicates = df_clean.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicates}")

if duplicates > 0:
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    n_after = len(df_clean)
    print(f"Baris sebelum: {n_before} -> sesudah: {n_after} ({n_before - n_after} baris dibuang)")
else:
    print("Data bersih, nggak ada duplikat.")

print(f"Shape sekarang: {df_clean.shape}")

# %% id="prep_step4_outlier_detect"
# ===== STEP 4: Deteksi dan Penanganan Outlier =====
# (dilakuin sebelum encoding/standarisasi biar hasilnya lebih akurat)

# deteksi outlier pake IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper, ((series < lower) | (series > upper)).sum()

num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

print("Deteksi outlier (IQR method):")
print("-" * 60)
outlier_info = {}
for col in num_cols:
    lower, upper, n_out = detect_outliers_iqr(df_clean[col])
    if n_out > 0:
        outlier_info[col] = (lower, upper, n_out)
        print(f"  {col:30s} -> {n_out:3d} outliers (range valid: {lower:.2f} - {upper:.2f})")

if not outlier_info:
    print("  Nggak ada outlier yang terdeteksi.")

# %% id="prep_step4_outlier_visual"
# visualisasi outlier pake boxplot
outlier_cols = list(outlier_info.keys()) if outlier_info else num_cols[:6]
n_plots = len(outlier_cols)

if n_plots > 0:
    fig, axes = plt.subplots(1, min(n_plots, 4), figsize=(4 * min(n_plots, 4), 5))
    if n_plots == 1:
        axes = [axes]

    for i, col in enumerate(outlier_cols[:4]):
        sns.boxplot(y=df_clean[col], ax=axes[i], color='lightskyblue')
        axes[i].set_title(f'{col}')

    plt.suptitle('Boxplot Kolom dengan Outlier', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

# %% id="prep_step4_outlier_cap"
# capping outlier pake batas IQR (winsorizing)
# daripada dibuang, kita clip nilainya ke batas atas/bawah biar nggak kehilangan data
for col, (lower, upper, n_out) in outlier_info.items():
    before_min = df_clean[col].min()
    before_max = df_clean[col].max()
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    print(f"  {col}: range ({before_min:.2f}, {before_max:.2f}) -> ({df_clean[col].min():.2f}, {df_clean[col].max():.2f})")

print("\nOutlier udah di-cap ke batas IQR.")

# %% id="prep_step5_encoding"
# ===== STEP 5: Encoding Data Kategorikal =====

# kolom kategorikal yang perlu di-encode
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f"Kolom kategorikal: {cat_cols}")

# tampilkan unique values sebelum encoding
for col in cat_cols:
    print(f"\n  {col}: {df_clean[col].unique()} ({df_clean[col].nunique()} unique)")

# %% id="prep_step5_encode_apply"
# pake LabelEncoder karena semua kolom kategorikal cuma punya 2 nilai (binary)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {mapping}")

print(f"\nSemua kolom kategorikal udah di-encode ke numerik.")
df_clean.dtypes

# %% id="prep_step6_binning"
# ===== STEP 6: Binning (Pengelompokan Data) =====

# bikin kategori umur buat analisis tambahan
# bins: Anak (<18), Dewasa Muda (18-35), Dewasa (36-55), Lansia (>55)
age_bins = [0, 18, 35, 55, 100]
age_labels = ['Anak', 'Dewasa_Muda', 'Dewasa', 'Lansia']

# binning berdasarkan nilai asli (sebelum standarisasi)
# kita pake kolom Age di dataframe original buat referensi
df_clean['Age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# lihat distribusi per group
print("Distribusi kelompok umur:")
print(df_clean['Age_group'].value_counts().sort_index())

# visualisasi sleep efficiency per age group
plt.figure(figsize=(8, 5))
sns.boxplot(x='Age_group', y='Sleep efficiency', data=df_clean, palette='viridis',
            order=age_labels)
plt.title('Sleep Efficiency per Kelompok Umur')
plt.xlabel('Kelompok Umur')
plt.ylabel('Sleep Efficiency')
plt.tight_layout()
plt.show()

# %% id="prep_step6_encode_bins"
# encode age group ke numerik juga, terus drop kolom category-nya
le_age = LabelEncoder()
df_clean['Age_group'] = le_age.fit_transform(df_clean['Age_group'])
print(f"Age group mapping: {dict(zip(le_age.classes_, le_age.transform(le_age.classes_)))}")

# %% id="prep_step3_standarisasi"
# ===== STEP 3: Normalisasi / Standarisasi Fitur =====
# dilakuin terakhir setelah encoding dan binning biar semua fitur numerik ikut ke-scale

target_col = 'Sleep efficiency'
feature_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col != target_col]

print(f"Fitur yang bakal distandarisasi ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  {col}: mean={df_clean[col].mean():.3f}, std={df_clean[col].std():.3f}")

scaler = StandardScaler()
df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

print(f"\nSetelah standarisasi:")
for col in feature_cols[:3]:
    print(f"  {col}: mean={df_clean[col].mean():.6f}, std={df_clean[col].std():.6f}")
print("  ...")

# %% id="prep_summary"
# ===== RINGKASAN PREPROCESSING =====

print("=" * 50)
print("RINGKASAN PREPROCESSING")
print("=" * 50)
print(f"Baris awal       : {len(df)}")
print(f"Baris akhir       : {len(df_clean)}")
print(f"Kolom awal        : {len(df.columns)}")
print(f"Kolom akhir       : {len(df_clean.columns)}")
print(f"Missing values    : 0")
print(f"Duplikat          : 0")
print(f"Target kolom      : {target_col}")
print(f"Fitur kolom       : {len(feature_cols)}")
print("-" * 50)
print(f"Kolom final: {list(df_clean.columns)}")

df_clean.head()

# %% id="prep_save"
# simpan dataset yang udah clean ke CSV
output_path = 'preprocessing/sleep_efficiency_preprocessing.csv'
df_clean.to_csv(output_path, index=False)
print(f"Dataset preprocessing disimpan ke: {output_path}")
print(f"Ukuran akhir: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom")
print("Siap dipake buat training model!")

