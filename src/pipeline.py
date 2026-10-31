# 1. IMPORT LIBRARY
import os
import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 2. LOKASI FILE
INPUT_PATH = "data/raw/automobileEDA_dirty_training.csv"
OUTPUT_PATH = "data/processed/automobileEDA_processed.csv"

# 3. FUNGSI MEMBACA DATA
def load_data(file_path):
    """Membaca dataset CSV menjadi DataFrame."""

    df = pd.read_csv(file_path)

    print("Dataset berhasil dibaca.")
    print("Lokasi dataset:", file_path)

    return df

# 4. FUNGSI INSPEKSI DATA
def inspect_data(df):
    """Memeriksa kondisi awal dataset."""

    print("\n=== LIMA BARIS PERTAMA ===")
    print(df.head().to_string(index=False))

    print("\n=== UKURAN DATASET ===")
    print("Jumlah baris:", df.shape[0])
    print("Jumlah kolom:", df.shape[1])

    print("\n=== NAMA KOLOM ===")
    print(df.columns.tolist())

    print("\n=== TIPE DATA ===")
    print(df.dtypes)

    print("\n=== MISSING VALUE ===")
    print(df.isna().sum())

    print("\n=== JUMLAH DUPLIKAT ===")
    print(df.duplicated().sum())

    categorical_columns = [
        "make",
        "num-of-doors",
        "body-style",
        "drive-wheels",
        "num-of-cylinders",
        "fuel-system",
        "horsepower-binned",
    ]

    print("\n=== NILAI UNIK KOLOM KATEGORI ===")

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].unique())

# 5. FUNGSI MENGUBAH FORMAT TANGGAL
def parse_transaction_date(value):
    """Mengubah beberapa format tanggal menjadi datetime."""

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        date_format = "%Y-%m-%d"

    elif re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        date_format = "%d/%m/%Y"

    elif re.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{4}", value):
        date_format = "%d-%b-%Y"

    elif re.fullmatch(r"\d{2}-\d{2}-\d{4}", value):
        date_format = "%m-%d-%Y"

    else:
        return pd.NaT

    return pd.to_datetime(
        value,
        format=date_format,
        errors="coerce",
    )

# 6. FUNGSI MEMBERSIHKAN DATA
def clean_data(df):
    """Membersihkan duplikat, teks, tanggal, dan missing value."""

    # Membuat salinan agar DataFrame awal tidak berubah
    df_clean = df.copy()

    rows_before = df_clean.shape[0]
    duplicates_before = df_clean.duplicated().sum()
    missing_before = df_clean.isna().sum().sum()

    # Menghapus baris duplikat
    df_clean = df_clean.drop_duplicates()

    # Mengambil semua kolom bertipe teks kecuali tanggal
    text_columns = df_clean.select_dtypes(
        include=["object", "string"]
    ).columns

    text_columns = [
        column
        for column in text_columns
        if column != "transaction_date"
    ]

    # Menghapus spasi dan menyeragamkan menjadi huruf kecil
    for column in text_columns:
        df_clean[column] = (
            df_clean[column]
            .str.strip()
            .str.lower()
        )

    # Menyeragamkan format tanggal
    df_clean["transaction_date"] = (
        df_clean["transaction_date"]
        .apply(parse_transaction_date)
    )

    # Mengisi kolom numerik dengan median
    numeric_missing_columns = [
        "stroke",
        "horsepower",
        "price",
    ]

    for column in numeric_missing_columns:
        median_value = df_clean[column].median()

        df_clean[column] = (
            df_clean[column]
            .fillna(median_value)
        )

    # Mengisi kolom kategori dengan modus
    categorical_missing_columns = [
        "make",
        "num-of-doors",
        "horsepower-binned",
    ]

    for column in categorical_missing_columns:
        mode_value = df_clean[column].mode()[0]

        df_clean[column] = (
            df_clean[column]
            .fillna(mode_value)
        )

    # Menghapus baris yang tidak memiliki tanggal transaksi
    df_clean = df_clean.dropna(
        subset=["transaction_date"]
    )

    # Mengatur ulang nomor indeks
    df_clean = df_clean.reset_index(drop=True)

    rows_after = df_clean.shape[0]
    missing_after = df_clean.isna().sum().sum()

    print("\n=== HASIL DATA CLEANING ===")
    print("Jumlah baris sebelum:", rows_before)
    print("Jumlah duplikat yang ditemukan:", duplicates_before)
    print("Jumlah missing value sebelum:", missing_before)
    print("Jumlah baris sesudah:", rows_after)
    print("Jumlah missing value sesudah:", missing_after)
    print("Jumlah duplikat sesudah:", df_clean.duplicated().sum())

    return df_clean

# 7. FUNGSI TRANSFORMASI DATA
def transform_data(df):
    """Melakukan scaling numerik dan encoding kategorikal."""

    df_transformed = df.copy()

    print("\n=== DATA SEBELUM TRANSFORMASI ===")
    print(
        df_transformed[
            ["price", "horsepower", "drive-wheels"]
        ]
        .head()
        .to_string(index=False)
    )

    # Min-Max Scaling
    scaler = MinMaxScaler()

    columns_to_scale = [
        "price",
        "horsepower",
    ]

    df_transformed[
        ["price_scaled", "horsepower_scaled"]
    ] = scaler.fit_transform(
        df_transformed[columns_to_scale]
    )

    # One-Hot Encoding
    df_transformed = pd.get_dummies(
        df_transformed,
        columns=["drive-wheels"],
        prefix="drive_wheels",
        dtype=int,
    )

    transformed_columns = [
        "price",
        "price_scaled",
        "horsepower",
        "horsepower_scaled",
        "drive_wheels_4wd",
        "drive_wheels_awd",
        "drive_wheels_fwd",
        "drive_wheels_rwd",
    ]

    print("\n=== DATA SETELAH TRANSFORMASI ===")
    print(
        df_transformed[transformed_columns]
        .head()
        .to_string(index=False)
    )

    return df_transformed

# 8. FUNGSI MENYIMPAN DATA
def save_data(df, output_path):
    """Menyimpan DataFrame yang sudah diproses menjadi CSV."""

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\n=== PENYIMPANAN DATA ===")
    print("Dataset berhasil disimpan.")
    print("Lokasi output:", output_path)
    print("Ukuran output:", df.shape)

# 9. URUTAN UTAMA PIPELINE
def main():
    # Extract dan load
    df = load_data(INPUT_PATH)

    # Inspect
    inspect_data(df)

    # Clean
    df_clean = clean_data(df)

    # Transform
    df_processed = transform_data(df_clean)

    # Save
    save_data(df_processed, OUTPUT_PATH)

# 10. MENJALANKAN PIPELINE
if __name__ == "__main__":
    main()