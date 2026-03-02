"""
automate_Rijal.py
Script otomatis buat preprocessing dataset Sleep Efficiency.
Ini hasil konversi dari notebook eksperimen ke file Python.
Outputnya berupa file CSV yang udah siap buat dilatih model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_data(filepath):
    """
    Muat dataset dari file CSV.
    Cek dulu filenya ada atau nggak biar nggak error.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} nggak ketemu, cek lagi pathnya.")

    df = pd.read_csv(filepath)
    print(f"Dataset berhasil dimuat. Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
    return df


def exploratory_analysis(df):
    """
    Cetak info dasar dataset buat analisis awal.
    Ini cuma ringkasan, visualisasi lengkapnya ada di notebook.
    """
    print("\n--- Info Dataset ---")
    print(df.dtypes)
    print(f"\nShape: {df.shape}")

    print("\n--- Statistik Deskriptif ---")
    print(df.describe())

    print("\n--- Jumlah Missing Values per Kolom ---")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = (count / len(df)) * 100
        print(f"  {col:30s} -> {count:3d} missing ({pct:.1f}%)")

    print(f"\n--- Jumlah Duplikat: {df.duplicated().sum()} ---")

    return df


def drop_irrelevant_columns(df):
    """
    Buang kolom yang nggak relevan buat modelling.
    ID cuma identifier, Bedtime dan Wakeup time formatnya datetime string.
    """
    cols_to_drop = ['ID', 'Bedtime', 'Wakeup time']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Kolom yang dibuang: {cols_to_drop}")
    print(f"Sisa kolom ({len(df.columns)}): {list(df.columns)}")
    return df


def handle_missing_values(df):
    """
    Step 1: Tangani missing values.
    Strategi: median buat numerik (lebih robust terhadap outlier), mode buat kategorikal.
    """
    print("\n===== STEP 1: Menangani Missing Values =====")

    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  [MEDIAN] '{col}' -> {null_count} missing diisi dengan {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  [MODE]   '{col}' -> {null_count} missing diisi dengan '{mode_val}'")

    print(f"  Sisa missing values: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    """
    Step 2: Hapus baris duplikat.
    """
    print("\n===== STEP 2: Menghapus Duplikat =====")

    n_before = len(df)
    duplicates = df.duplicated().sum()
    print(f"  Jumlah duplikat: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  Baris sebelum: {n_before} -> sesudah: {len(df)}")
    else:
        print("  Data bersih, nggak ada duplikat.")

    return df


def handle_outliers(df):
    """
    Step 4: Deteksi dan penanganan outlier pake IQR method.
    Strategi: capping (winsorizing) ke batas IQR, biar nggak kehilangan data.
    Dilakuin sebelum encoding biar nggak kena kolom kategorikal.
    """
    print("\n===== STEP 4: Deteksi & Penanganan Outlier =====")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if n_outliers > 0:
            before_min = df[col].min()
            before_max = df[col].max()
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"  {col}: {n_outliers} outliers, "
                  f"range ({before_min:.2f}, {before_max:.2f}) -> ({df[col].min():.2f}, {df[col].max():.2f})")

    print("  Outlier udah di-cap ke batas IQR.")
    return df


def encode_categorical(df):
    """
    Step 5: Encode kolom kategorikal pake LabelEncoder.
    Semua kolom kategorikal di dataset ini binary (2 nilai), jadi LabelEncoder cukup.
    """
    print("\n===== STEP 5: Encoding Data Kategorikal =====")

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  {col}: {mapping}")

    return df, label_encoders


def binning_age(df, df_original):
    """
    Step 6: Binning umur ke kelompok kategori.
    Pake data umur asli (sebelum standarisasi) buat bikin bins yang akurat.
    """
    print("\n===== STEP 6: Binning (Pengelompokan Data) =====")

    age_bins = [0, 18, 35, 55, 100]
    age_labels = ['Anak', 'Dewasa_Muda', 'Dewasa', 'Lansia']

    # pake kolom Age dari data original buat referensi binning
    df['Age_group'] = pd.cut(df_original['Age'], bins=age_bins, labels=age_labels, right=False)

    print("  Distribusi kelompok umur:")
    for group, count in df['Age_group'].value_counts().sort_index().items():
        print(f"    {group}: {count}")

    # encode age group ke numerik
    le_age = LabelEncoder()
    df['Age_group'] = le_age.fit_transform(df['Age_group'])
    mapping = dict(zip(le_age.classes_, le_age.transform(le_age.classes_)))
    print(f"  Age group mapping: {mapping}")

    return df


def standardize_features(df, target_col='Sleep efficiency'):
    """
    Step 3: Standarisasi fitur numerik (kecuali target).
    Dilakuin terakhir setelah semua transformasi biar semua fitur ikut ke-scale.
    """
    print("\n===== STEP 3: Standarisasi Fitur =====")

    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"  {len(feature_cols)} fitur distandarisasi: {feature_cols}")

    return df, scaler


def save_data(df, output_path):
    """
    Simpan dataset yang udah dipreprocessing ke file CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"\nDataset preprocessing disimpan ke: {output_path}")
    print(f"Ukuran akhir: {df.shape[0]} baris, {df.shape[1]} kolom")


def main():
    """
    Fungsi utama buat jalanin semua tahapan preprocessing sekaligus.
    Urutan: Load -> EDA -> Drop kolom -> Missing Values -> Duplikat ->
            Outlier -> Encoding -> Binning -> Standarisasi -> Simpan
    """
    # path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, '..', 'sleep_efficiency_raw.csv')
    output_path = os.path.join(script_dir, 'sleep_efficiency_preprocessing.csv')

    print("=" * 60)
    print("PREPROCESSING OTOMATIS - Sleep Efficiency Dataset")
    print("=" * 60)

    # muat data
    df = load_data(raw_data_path)
    df_original = df.copy()

    # analisis awal
    df = exploratory_analysis(df)

    # persiapan: buang kolom nggak relevan
    df = drop_irrelevant_columns(df)

    # step 1: tangani missing values
    df = handle_missing_values(df)

    # step 2: hapus duplikat
    df = remove_duplicates(df)

    # step 4: deteksi dan cap outlier (sebelum encoding)
    df = handle_outliers(df)

    # step 5: encode kolom kategorikal
    df, encoders = encode_categorical(df)

    # step 6: binning umur
    df = binning_age(df, df_original)

    # step 3: standarisasi (terakhir, setelah semua transformasi)
    df, scaler = standardize_features(df)

    # ringkasan
    print("\n" + "=" * 50)
    print("RINGKASAN PREPROCESSING")
    print("=" * 50)
    print(f"Baris awal        : {len(df_original)}")
    print(f"Baris akhir       : {len(df)}")
    print(f"Kolom awal        : {len(df_original.columns)}")
    print(f"Kolom akhir       : {len(df.columns)}")
    print(f"Missing values    : 0")
    print(f"Duplikat          : 0")
    print(f"Kolom final       : {list(df.columns)}")

    # simpan
    save_data(df, output_path)
    print("\nSelesai! Dataset siap dipake buat training model.")


if __name__ == '__main__':
    main()
