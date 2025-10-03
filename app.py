# ===================================================================================
# IMPORT PUSTAKA YANG DIBUTUHKAN
# ===================================================================================
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
# Pustaka baru untuk mempermudah penulisan ke Google Sheets
from gspread_dataframe import set_with_dataframe


# ===================================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ===================================================================================
st.set_page_config(
    page_title="Dashboard Analisis Kompetitor",
    page_icon="🧠",
    layout="wide"
)


# ===================================================================================
# FUNGSI-FUNGSI UTAMA (Backend Logic)
# ===================================================================================

# --- FUNGSI KONEKSI DAN PEMUATAN DATA ---
@st.cache_resource(ttl=600)
def connect_to_gsheets():
    """
    Membuat koneksi ke Google Sheets menggunakan kredensial dari st.secrets.
    Menggunakan cache untuk menghindari koneksi berulang.
    """
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Gagal terhubung ke Google API: {e}")
        return None

@st.cache_data(ttl=600)
def load_and_process_data(_client):
    """
    Menarik semua data dari worksheet di Google Sheets dan memprosesnya.
    Menggunakan cache data untuk menghindari penarikan data berulang.
    """
    try:
        spreadsheet = _client.open_by_key(st.secrets["SOURCE_SPREADSHEET_ID"])
        worksheets = spreadsheet.worksheets()
        
        all_dfs = {}
        for ws in worksheets:
            # Mengambil semua data dan membuat DataFrame
            data = ws.get_all_records()
            if not data: continue # Lewati worksheet kosong
            df = pd.DataFrame(data)
            # Membersihkan nama kolom dari spasi ekstra
            df.columns = [col.strip() for col in df.columns]
            all_dfs[ws.title] = df

        # --- Penggabungan dan Pembersihan Data ---
        # 1. Normalisasi Brand
        df_kamus = all_dfs.get("kamus_brand", pd.DataFrame())
        if not df_kamus.empty:
            kamus_brand_dict = dict(zip(df_kamus['Alias'], df_kamus['Brand_Utama']))
        else:
            kamus_brand_dict = {}

        # 2. Proses semua sheet REKAP
        list_df_kompetitor = []
        for title, df in all_dfs.items():
            if 'REKAP' in title and 'DB KLIK' not in title:
                toko = title.split(' - ')[0].strip()
                df['TOKO'] = toko
                df['STATUS_STOK'] = 'Ready' if 'READY' in title else 'Habis'
                list_df_kompetitor.append(df)

        df_kompetitor_gabungan = pd.concat(list_df_kompetitor, ignore_index=True) if list_df_kompetitor else pd.DataFrame()
        
        # 3. Proses sheet DB KLIK
        df_db_klik_ready = all_dfs.get("DB KLIK - REKAP - READY", pd.DataFrame())
        df_db_klik_habis = all_dfs.get("DB KLIK - REKAP - HABIS", pd.DataFrame())
        df_db_klik_ready['STATUS_STOK'] = 'Ready'
        df_db_klik_habis['STATUS_STOK'] = 'Habis'
        df_db_klik = pd.concat([df_db_klik_ready, df_db_klik_habis], ignore_index=True)

        # 4. Proses sheet DATABASE
        df_database = all_dfs.get("DATABASE", pd.DataFrame())
        
        # --- Konversi Tipe Data ---
        for df in [df_kompetitor_gabungan, df_db_klik, df_database]:
            if df.empty: continue
            if 'HARGA' in df.columns:
                df['HARGA'] = pd.to_numeric(df['HARGA'], errors='coerce').fillna(0)
            if 'TERJUAL/BLN' in df.columns:
                df['TERJUAL/BLN'] = pd.to_numeric(df['TERJUAL/BLN'], errors='coerce').fillna(0)
            if 'TANGGAL' in df.columns:
                 # [OPTIMASI] Tambahkan dayfirst=True untuk handle format DD/MM/YYYY dan menghilangkan warning
                 df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce', dayfirst=True)
            # Normalisasi brand
            if 'BRAND' in df.columns:
                df['BRAND'] = df['BRAND'].str.strip().str.upper()
                df['BRAND'] = df['BRAND'].replace(kamus_brand_dict)


        return df_kompetitor_gabungan, df_db_klik, df_database

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Error: Spreadsheet tidak ditemukan. Periksa kembali `SOURCE_SPREADSHEET_ID` di secrets Anda.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data dari Google Sheets: {e}")
        st.info("Pastikan konfigurasi `secrets.toml` sudah benar dan koneksi internet stabil.")
        return None, None, None


# --- [OPTIMASI] FUNGSI PELABELAN YANG LEBIH RINGAN ---
def perform_labeling(df_db_klik, df_database, progress_placeholder):
    """
    Melakukan pelabelan SKU dan KATEGORI secara efisien dengan memproses per merek.
    Ini mengurangi beban komputasi secara drastis.
    """
    # Persiapan awal: pastikan tipe data benar dan tangani brand kosong
    df_database['NAMA'] = df_database['NAMA'].astype(str)
    df_database['SKU'] = df_database['SKU'].astype(str)
    df_db_klik['NAMA'] = df_db_klik['NAMA'].astype(str)
    
    df_db_klik['BRAND'].fillna('TIDAK_DIKETAHUI', inplace=True)
    df_database['BRAND'].fillna('TIDAK_DIKETAHUI', inplace=True)

    unique_brands = df_db_klik['BRAND'].unique()
    labeled_dfs = []
    
    # Setup progress bar
    progress_bar = progress_placeholder.progress(0, text="Memulai proses pelabelan...")

    # Loop untuk setiap merek, ini adalah inti dari optimasi
    for i, brand in enumerate(unique_brands):
        progress_text = f"Memproses brand: {brand} ({i+1}/{len(unique_brands)})"
        progress_bar.progress((i + 1) / len(unique_brands), text=progress_text)

        # Filter data hanya untuk merek saat ini
        db_klik_brand_df = df_db_klik[df_db_klik['BRAND'] == brand].copy()
        database_brand_df = df_database[df_database['BRAND'] == brand].copy()

        # Jika tidak ada produk dari merek ini di database, beri label khusus dan lanjutkan
        if database_brand_df.empty:
            db_klik_brand_df['SKU'] = 'TIDAK_DITEMUKAN_DI_DB'
            db_klik_brand_df['KATEGORI'] = 'TIDAK_DITEMUKAN_DI_DB'
            labeled_dfs.append(db_klik_brand_df)
            continue
        
        # Lakukan TF-IDF hanya pada data yang sudah difilter (jauh lebih ringan)
        database_brand_df['text_for_tfidf'] = database_brand_df['NAMA'].fillna('') + ' ' + database_brand_df['SKU'].fillna('')
        db_klik_brand_df['text_for_tfidf'] = db_klik_brand_df['NAMA'].fillna('')

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_database = vectorizer.fit_transform(database_brand_df['text_for_tfidf'])
        tfidf_db_klik = vectorizer.transform(db_klik_brand_df['text_for_tfidf'])

        cosine_similarities = cosine_similarity(tfidf_db_klik, tfidf_database)
        best_matches_indices = cosine_similarities.argmax(axis=1)

        # Ambil SKU dan Kategori dari hasil pencocokan
        matched_skus = database_brand_df.iloc[best_matches_indices]['SKU'].values
        matched_kategoris = database_brand_df.iloc[best_matches_indices]['KATEGORI'].values

        db_klik_brand_df['SKU'] = matched_skus
        db_klik_brand_df['KATEGORI'] = matched_kategoris
        
        labeled_dfs.append(db_klik_brand_df)
    
    progress_placeholder.empty() # Hapus progress bar setelah selesai

    # Gabungkan semua hasil pelabelan dari setiap merek
    if not labeled_dfs:
        return df_db_klik

    final_labeled_df = pd.concat(labeled_dfs).sort_index()
    
    if 'text_for_tfidf' in final_labeled_df.columns:
        final_labeled_df.drop(columns=['text_for_tfidf'], inplace=True)
        
    return final_labeled_df

# --- FUNGSI UNTUK MENULIS KEMBALI KE GOOGLE SHEETS ---
def update_spreadsheet_with_labels(client, labeled_df):
    """
    Menyimpan kembali data DB KLIK yang sudah dilabeli ke worksheet yang sesuai.
    """
    try:
        spreadsheet = client.open_by_key(st.secrets["SOURCE_SPREADSHEET_ID"])
        
        # Pisahkan kembali data READY dan HABIS
        df_ready = labeled_df[labeled_df['STATUS_STOK'] == 'Ready'].copy()
        df_habis = labeled_df[labeled_df['STATUS_STOK'] == 'Habis'].copy()
        
        # Tentukan kolom yang akan ditulis (sesuai urutan di GSheet)
        target_columns = ['TANGGAL', 'NAMA', 'HARGA', 'TERJUAL/BLN', 'STOK', 'BRAND', 'KATEGORI', 'SKU']
        
        # Pastikan kolom tanggal berformat string agar tidak error saat ditulis
        df_ready['TANGGAL'] = pd.to_datetime(df_ready['TANGGAL'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_habis['TANGGAL'] = pd.to_datetime(df_habis['TANGGAL'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Menulis ke worksheet READY
        ws_ready = spreadsheet.worksheet("DB KLIK - REKAP - READY")
        ws_ready.clear() 
        set_with_dataframe(ws_ready, df_ready[target_columns].fillna(''), include_index=False, resize=True)
        
        # Menulis ke worksheet HABIS
        ws_habis = spreadsheet.worksheet("DB KLIK - REKAP - HABIS")
        ws_habis.clear() 
        set_with_dataframe(ws_habis, df_habis[target_columns].fillna(''), include_index=False, resize=True)

        return True
    except Exception as e:
        st.error(f"Gagal menyimpan hasil pelabelan ke Google Sheets: {e}")
        st.warning("Pastikan email service account memiliki akses 'Editor' pada file Google Sheets Anda.")
        return False
        
# --- FUNGSI UNTUK MENCARI KECOCOKAN PRODUK ANTAR TOKO (TF-IDF) ---
def find_product_matches_tfidf(selected_product_name, df_db_klik, df_kompetitor):
    """
    Mencari produk yang mirip dari kompetitor menggunakan TF-IDF.
    """
    selected_product = df_db_klik[df_db_klik['NAMA'] == selected_product_name]
    if selected_product.empty: return []
    
    selected_brand = selected_product['BRAND'].iloc[0]
    
    competitor_filtered_by_brand = df_kompetitor[df_kompetitor['BRAND'] == selected_brand].copy()
    if competitor_filtered_by_brand.empty:
        db_klik_info = selected_product.iloc[0].to_dict()
        db_klik_info['TOKO'] = 'DB KLIK'
        db_klik_info['SKOR_KEMIRIPAN'] = 100
        return [db_klik_info]

    combined_df = pd.concat([selected_product, competitor_filtered_by_brand], ignore_index=True)
    
    combined_df['NAMA_CLEAN'] = combined_df['NAMA'].fillna('').astype(str).str.lower()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(combined_df['NAMA_CLEAN'])
    
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
    
    matches = []
    db_klik_info = selected_product.iloc[0].to_dict()
    db_klik_info['TOKO'] = 'DB KLIK'
    db_klik_info['SKOR_KEMIRIPAN'] = 100
    matches.append(db_klik_info)

    threshold = 0.50
    for i in range(1, len(cosine_sim)):
        if cosine_sim[i] >= threshold:
            match_info = combined_df.iloc[i].to_dict()
            match_info['SKOR_KEMIRIPAN'] = round(cosine_sim[i] * 100)
            matches.append(match_info)
            
    return matches


# --- FUNGSI UNTUK MENGUBAH DF KE EXCEL ---
def to_excel(df_dict):
    """
    Mengekspor beberapa DataFrame ke dalam satu file Excel dengan sheet berbeda.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df_copy = df.copy()
            for col in df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
            df_copy.to_excel(writer, index=False, sheet_name=sheet_name)
    processed_data = output.getvalue()
    return processed_data


# ===================================================================================
# TAMPILAN APLIKASI (Frontend UI)
# ===================================================================================

st.title("🧠 Dashboard Analisis Kompetitor Cerdas")
st.markdown("Selamat datang, Firman! Platform ini dirancang untuk memberikan analisis mendalam terhadap data penjualan DB KLIK dan para kompetitornya.")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_kompetitor = pd.DataFrame()
    st.session_state.df_db_klik = pd.DataFrame()
    st.session_state.df_database = pd.DataFrame()

if not st.session_state.data_loaded:
    with st.spinner("Menarik dan memproses data dari semua toko... Harap tunggu."):
        client = connect_to_gsheets()
        if client:
            df_kompetitor, df_db_klik, df_database = load_and_process_data(client)
            if df_kompetitor is not None and df_db_klik is not None and df_database is not None:
                st.session_state.df_kompetitor = df_kompetitor
                st.session_state.df_db_klik = df_db_klik
                st.session_state.df_database = df_database
                st.session_state.data_loaded = True
        else:
            st.error("Koneksi ke Google Sheets gagal. Aplikasi tidak dapat berjalan.")
            st.stop()

if st.session_state.data_loaded:
    needs_labeling = st.session_state.df_db_klik['SKU'].isnull().any() or \
                     st.session_state.df_db_klik['KATEGORI'].isnull().any() or \
                     (st.session_state.df_db_klik['SKU'] == '').any() or \
                     (st.session_state.df_db_klik['KATEGORI'] == '').any()

    if needs_labeling:
        st.warning("⚠️ **PERHATIAN:** Terdeteksi pelabelan SKU dan KATEGORI tidak sinkron atau data tidak ditemukan. Silakan jalankan proses pelabelan.")
        if st.button("JALANKAN PELABELAN SKU DAN KATEGORI"):
            # [OPTIMASI] Buat placeholder untuk progress bar
            progress_placeholder = st.empty()
            with st.spinner("Mempersiapkan pelabelan cerdas..."):
                # Panggil fungsi pelabelan yang sudah dioptimasi
                df_db_klik_labeled = perform_labeling(
                    st.session_state.df_db_klik.copy(), 
                    st.session_state.df_database.copy(),
                    progress_placeholder
                )
                
                client = connect_to_gsheets()
                success = update_spreadsheet_with_labels(client, df_db_klik_labeled)
                
                if success:
                    st.success("Pelabelan selesai dan data telah berhasil disimpan kembali ke Google Sheets!")
                    st.cache_data.clear()
                    st.info("Memuat ulang aplikasi dengan data terbaru...")
                    st.session_state.data_loaded = False
                    st.rerun()
                else:
                    st.error("Proses pelabelan gagal disimpan. Silakan periksa error di atas.")
        st.stop()
else:
    st.error("Gagal memuat data. Tidak dapat melanjutkan.")
    st.stop()

if st.session_state.df_db_klik.empty or st.session_state.df_kompetitor.empty:
    st.error("Data DB KLIK atau Kompetitor kosong. Tidak dapat melanjutkan analisis.")
    st.stop()

# ===================================================================================
# SIDEBAR UNTUK FILTER DAN NAVIGASI
# ===================================================================================
with st.sidebar:
    st.header("Filter & Navigasi")
    
    analysis_mode = st.radio(
        "Pilih Mode Analisis:",
        ("ANALISIS UTAMA", "HPP PRODUK", "SIMILARITY PRODUK")
    )
    
    st.divider()

    df_gabungan = pd.concat([st.session_state.df_db_klik, st.session_state.df_kompetitor], ignore_index=True)
    df_gabungan['TANGGAL'] = pd.to_datetime(df_gabungan['TANGGAL'], errors='coerce')
    df_gabungan.dropna(subset=['TANGGAL'], inplace=True)

    min_date = df_gabungan['TANGGAL'].min().date()
    max_date = df_gabungan['TANGGAL'].max().date()
    
    st.info(f"Data tersedia dari:\n\n**{min_date.strftime('%d %B %Y')}** hingga **{max_date.strftime('%d %B %Y')}**")

    date_range = st.date_input(
        "Pilih Rentang Waktu Analisis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_filter"
    )
    
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)
        
    mask_db_klik = (st.session_state.df_db_klik['TANGGAL'] >= start_date) & (st.session_state.df_db_klik['TANGGAL'] <= end_date)
    df_db_klik_filtered = st.session_state.df_db_klik[mask_db_klik].copy()
    
    mask_kompetitor = (st.session_state.df_kompetitor['TANGGAL'] >= start_date) & (st.session_state.df_kompetitor['TANGGAL'] <= end_date)
    df_kompetitor_filtered = st.session_state.df_kompetitor[mask_kompetitor].copy()
    
    latest_date = df_gabungan['TANGGAL'].max()
    df_db_klik_latest = st.session_state.df_db_klik[st.session_state.df_db_klik['TANGGAL'] == latest_date].copy()
    df_kompetitor_latest = st.session_state.df_kompetitor[st.session_state.df_kompetitor['TANGGAL'] == latest_date].copy()

    # [OPTIMASI] Hitung kolom OMZET sekali saja
    df_db_klik_filtered['OMZET'] = df_db_klik_filtered['HARGA'] * df_db_klik_filtered['TERJUAL/BLN']
    df_kompetitor_filtered['OMZET'] = df_kompetitor_filtered['HARGA'] * df_kompetitor_filtered['TERJUAL/BLN']
    df_db_klik_latest['OMZET'] = df_db_klik_latest['HARGA'] * df_db_klik_latest['TERJUAL/BLN']

    st.divider()

    st.write("Butuh pelabelan ulang?")
    if st.button("Jalankan Ulang Pelabelan"):
        progress_placeholder_sidebar = st.empty()
        with st.spinner("Memaksa pelabelan ulang..."):
            df_db_klik_labeled = perform_labeling(st.session_state.df_db_klik.copy(), st.session_state.df_database.copy(), progress_placeholder_sidebar)
            client = connect_to_gsheets()
            success = update_spreadsheet_with_labels(client, df_db_klik_labeled)
            if success:
                st.success("Pelabelan ulang berhasil disimpan!")
                st.cache_data.clear()
                st.session_state.data_loaded = False
                st.rerun()
            else:
                st.error("Gagal menyimpan hasil pelabelan ulang.")
    
    st.divider()

    st.metric(label="Total Baris Data Dianalisis", value=f"{len(df_db_klik_filtered) + len(df_kompetitor_filtered):,}")

    df_to_export = {
        "DB_KLIK_Processed": df_db_klik_filtered,
        "KOMPETITOR_Processed": df_kompetitor_filtered,
        "DATABASE_Master": st.session_state.df_database
    }
    excel_data = to_excel(df_to_export)
    st.download_button(
        label="📥 Unduh Data Excel",
        data=excel_data,
        file_name=f"analisis_data_{start_date.date()}_{end_date.date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===================================================================================
# KONTEN UTAMA BERDASARKAN MODE ANALISIS
# ===================================================================================

if analysis_mode == "ANALISIS UTAMA":
    st.header("📊 Analisis Utama")
    
    tab1, tab2, tab3 = st.tabs(["Analisis DB KLIK", "Analisis Kompetitor", "Perbandingan Produk (Baru & Habis)"])

    with tab1:
        st.subheader("Performa Penjualan DB KLIK")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Peringkat Omzet per Kategori")
            if 'KATEGORI' in df_db_klik_filtered.columns and not df_db_klik_filtered['KATEGORI'].dropna().empty:
                omzet_per_kategori = df_db_klik_filtered.groupby('KATEGORI')['OMZET'].sum().sort_values(ascending=False).reset_index()
                omzet_per_kategori['OMZET_formatted'] = omzet_per_kategori['OMZET'].apply(lambda x: f"Rp {x:,.0f}")
                st.dataframe(omzet_per_kategori[['KATEGORI', 'OMZET_formatted']], use_container_width=True)
                kategori_pilihan = st.selectbox("Pilih Kategori:", omzet_per_kategori['KATEGORI'].unique())
            else:
                st.info("Kolom 'KATEGORI' kosong atau tidak ada.")
                kategori_pilihan = None
            
        with col2:
            st.markdown("#### Visualisasi Peringkat Kategori")
            if kategori_pilihan:
                num_bars = st.slider("Jumlah kategori:", 1, len(omzet_per_kategori), min(10, len(omzet_per_kategori)))
                sort_order = st.radio("Urutkan:", ('Tertinggi ke Terendah', 'Terendah ke Tertinggi'), horizontal=True, key="cat_sort")
                ascending = (sort_order == 'Terendah ke Tertinggi')
                top_kategori = omzet_per_kategori.sort_values('OMZET', ascending=ascending).head(num_bars)
                fig_kat = px.bar(top_kategori, x='OMZET', y='KATEGORI', orientation='h', text='OMZET_formatted')
                fig_kat.update_layout(yaxis={'categoryorder':'total ascending' if ascending else 'total descending'})
                st.plotly_chart(fig_kat, use_container_width=True)

        if kategori_pilihan:
            st.markdown(f"#### Produk Terlaris di Kategori: **{kategori_pilihan}** (Data Terbaru)")
            produk_kategori = df_db_klik_latest[df_db_klik_latest['KATEGORI'] == kategori_pilihan]
            produk_kategori = produk_kategori.sort_values('OMZET', ascending=False)[['NAMA', 'HARGA', 'TERJUAL/BLN', 'OMZET']]
            produk_kategori['HARGA'] = produk_kategori['HARGA'].apply(lambda x: f"Rp {x:,.0f}")
            produk_kategori['OMZET'] = produk_kategori['OMZET'].apply(lambda x: f"Rp {x:,.0f}")
            st.dataframe(produk_kategori, use_container_width=True)

        st.divider()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### 6 Brand dengan Omzet Tertinggi (Data Terbaru)")
            omzet_brand_latest = df_db_klik_latest.groupby('BRAND')['OMZET'].sum().nlargest(6)
            if not omzet_brand_latest.empty:
                fig_brand = px.pie(omzet_brand_latest, values='OMZET', names=omzet_brand_latest.index, hole=0.3)
                fig_brand.update_traces(textinfo='percent+label', texttemplate='%{label}<br>Rp %{value:,.0f}')
                st.plotly_chart(fig_brand, use_container_width=True)

            st.markdown("#### Ringkasan Performa Brand (Periode Dipilih)")
            brand_summary = df_db_klik_filtered.groupby('BRAND').agg(
                Total_Omzet=('OMZET', 'sum'),
                Total_Unit_Terjual=('TERJUAL/BLN', 'sum')
            ).sort_values('Total_Omzet', ascending=False).reset_index()
            brand_summary['Total_Omzet'] = brand_summary['Total_Omzet'].apply(lambda x: f"Rp {x:,.0f}")
            st.dataframe(brand_summary, use_container_width=True)

        with col4:
            st.markdown("#### Produk Terlaris Global (Periode Dipilih)")
            produk_terlaris = df_db_klik_filtered.sort_values('OMZET', ascending=False).head(15)
            produk_terlaris = produk_terlaris[['NAMA', 'SKU', 'HARGA', 'OMZET']]
            produk_terlaris['HARGA'] = produk_terlaris['HARGA'].apply(lambda x: f"Rp {x:,.0f}")
            produk_terlaris['OMZET'] = produk_terlaris['OMZET'].apply(lambda x: f"Rp {x:,.0f}")
            st.dataframe(produk_terlaris, use_container_width=True)

        st.divider()
        st.subheader("Analisis Pertumbuhan Week-over-Week (WoW)")
        df_db_klik_filtered['MINGGU'] = df_db_klik_filtered['TANGGAL'].dt.to_period('W').apply(lambda r: r.start_time).dt.date
        wow_summary = df_db_klik_filtered.groupby('MINGGU').agg(Omzet=('OMZET', 'sum'), Total_Unit_Terjual=('TERJUAL/BLN', 'sum')).sort_index()
        wow_summary['Omzet_Sebelumnya'] = wow_summary['Omzet'].shift(1)
        wow_summary['Persentase_Pertumbuhan'] = ((wow_summary['Omzet'] - wow_summary['Omzet_Sebelumnya']) / wow_summary['Omzet_Sebelumnya']) * 100
        wow_summary.fillna(0, inplace=True)
        wow_summary_display = wow_summary[['Omzet', 'Total_Unit_Terjual', 'Persentase_Pertumbuhan']].copy()
        wow_summary_display['Omzet'] = wow_summary_display['Omzet'].apply(lambda x: f"Rp {x:,.0f}")
        wow_summary_display['Persentase_Pertumbuhan'] = wow_summary_display['Persentase_Pertumbuhan'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(wow_summary_display.reset_index(), use_container_width=True)

    with tab2:
        st.subheader("Performa Penjualan Kompetitor")
        st.markdown("#### Analisis Brand Kompetitor (Periode Dipilih)")
        unique_toko = df_kompetitor_filtered['TOKO'].unique()
        if len(unique_toko) > 0:
            toko_pilihan = st.selectbox("Pilih Toko Kompetitor:", unique_toko)
            df_toko_terpilih = df_kompetitor_filtered[df_kompetitor_filtered['TOKO'] == toko_pilihan]
            
            col5, col6 = st.columns(2)
            with col5:
                st.markdown(f"##### 6 Brand Omzet Tertinggi di **{toko_pilihan}**")
                omzet_brand_kompetitor = df_toko_terpilih.groupby('BRAND')['OMZET'].sum().nlargest(6)
                if not omzet_brand_kompetitor.empty:
                    fig_brand_komp = px.pie(omzet_brand_kompetitor, values='OMZET', names=omzet_brand_kompetitor.index, hole=0.3)
                    fig_brand_komp.update_traces(textinfo='percent+label', texttemplate='%{label}<br>Rp %{value:,.0f}')
                    st.plotly_chart(fig_brand_komp, use_container_width=True)
                else:
                    st.info("Tidak ada data omzet untuk toko ini pada periode terpilih.")
            
            with col6:
                st.markdown(f"##### Ringkasan Performa Brand di **{toko_pilihan}**")
                brand_summary_komp = df_toko_terpilih.groupby('BRAND').agg(Total_Omzet=('OMZET', 'sum'), Total_Unit_Terjual=('TERJUAL/BLN', 'sum')).sort_values('Total_Omzet', ascending=False).reset_index()
                brand_summary_komp['Total_Omzet'] = brand_summary_komp['Total_Omzet'].apply(lambda x: f"Rp {x:,.0f}")
                st.dataframe(brand_summary_komp, height=350, use_container_width=True)
        else:
            st.info("Tidak ada data kompetitor pada rentang tanggal yang dipilih.")
            
        st.divider()

        st.markdown("#### Perbandingan Pendapatan Antar Toko")
        df_db_klik_filtered['TOKO'] = 'DB KLIK'
        df_all_stores_filtered = pd.concat([df_db_klik_filtered, df_kompetitor_filtered], ignore_index=True)
        
        pendapatan_per_hari = df_all_stores_filtered.groupby(['TANGGAL', 'TOKO'])['OMZET'].sum().reset_index()
        fig_pendapatan = px.line(pendapatan_per_hari, x='TANGGAL', y='OMZET', color='TOKO', title="Tren Pendapatan Harian Semua Toko")
        st.plotly_chart(fig_pendapatan, use_container_width=True)

        pendapatan_pivot = pendapatan_per_hari.pivot_table(index='TOKO', columns='TANGGAL', values='OMZET', fill_value=0)
        pendapatan_pivot.columns = pendapatan_pivot.columns.strftime('%Y-%m-%d')
        pendapatan_pivot = pendapatan_pivot.applymap(lambda x: f"Rp {x:,.0f}")
        st.dataframe(pendapatan_pivot)
        
        st.divider()
        
        st.markdown("#### Tren Ketersediaan Produk (Ready vs Habis)")
        status_counts = df_all_stores_filtered.groupby(['TANGGAL', 'TOKO', 'STATUS_STOK']).size().reset_index(name='JUMLAH_PRODUK')
        fig_status = px.line(status_counts, x='TANGGAL', y='JUMLAH_PRODUK', color='TOKO', line_dash='STATUS_STOK', title="Jumlah Produk Ready vs Habis")
        st.plotly_chart(fig_status, use_container_width=True)
        status_pivot = status_counts.pivot_table(index=['TANGGAL', 'TOKO'], columns='STATUS_STOK', values='JUMLAH_PRODUK', fill_value=0).reset_index()
        st.dataframe(status_pivot)
    
    with tab3:
        st.subheader("Perbandingan Snapshot Produk")
        st.write("Bandingkan daftar produk dari dua tanggal berbeda untuk melihat produk baru/hilang.")

        col7, col8 = st.columns(2)
        with col7:
            tanggal_pembanding = st.date_input("Pilih Tanggal Pembanding", value=min_date, min_value=min_date, max_value=max_date)
        with col8:
            tanggal_target = st.date_input("Pilih Tanggal Target", value=max_date, min_value=min_date, max_value=max_date)

        if tanggal_pembanding and tanggal_target:
            df_pembanding = df_gabungan[df_gabungan['TANGGAL'].dt.date == tanggal_pembanding]
            df_target = df_gabungan[df_gabungan['TANGGAL'].dt.date == tanggal_target]

            toko_list = sorted(list(df_gabungan['TOKO'].unique()))
            toko_list.insert(0, 'DB KLIK')
            toko_compare_pilihan = st.selectbox("Pilih Toko untuk Dibandingkan", toko_list)

            produk_pembanding = set(df_pembanding[df_pembanding['TOKO'] == toko_compare_pilihan]['NAMA'])
            produk_target = set(df_target[df_target['TOKO'] == toko_compare_pilihan]['NAMA'])

            produk_baru = produk_target - produk_pembanding
            produk_hilang = produk_pembanding - produk_target

            col9, col10 = st.columns(2)
            with col9:
                st.markdown(f"#### Produk Baru di **{toko_compare_pilihan}**")
                st.dataframe(pd.DataFrame(list(produk_baru), columns=['Nama Produk']), use_container_width=True)
            with col10:
                st.markdown(f"#### Produk Hilang/Habis di **{toko_compare_pilihan}**")
                st.dataframe(pd.DataFrame(list(produk_hilang), columns=['Nama Produk']), use_container_width=True)

elif analysis_mode == "HPP PRODUK":
    st.header("📦 Analisis Harga Pokok Penjualan (HPP) Produk DB KLIK")
    
    if 'HPP (LATEST)' in st.session_state.df_database.columns:
        df_database_hpp = st.session_state.df_database[['SKU', 'HPP (LATEST)']].copy()
        df_database_hpp.rename(columns={'HPP (LATEST)': 'HPP'}, inplace=True)
        df_database_hpp['HPP'] = pd.to_numeric(df_database_hpp['HPP'], errors='coerce').fillna(0)
        
        df_merged_hpp = pd.merge(df_db_klik_latest, df_database_hpp, on='SKU', how='left')
        df_merged_hpp['HPP'].fillna(0, inplace=True)
        df_merged_hpp = df_merged_hpp[df_merged_hpp['HPP'] > 0]
        
        df_merged_hpp['SELISIH_HPP'] = df_merged_hpp['HARGA'] - df_merged_hpp['HPP']

        produk_lebih_mahal = df_merged_hpp[df_merged_hpp['SELISIH_HPP'] > 0].sort_values('SELISIH_HPP', ascending=False)
        produk_lebih_murah = df_merged_hpp[df_merged_hpp['SELISIH_HPP'] < 0].sort_values('SELISIH_HPP', ascending=True)

        st.markdown("#### Produk dengan Harga Jual > HPP Terbaru")
        st.dataframe(produk_lebih_mahal[['NAMA', 'SKU', 'HARGA', 'HPP', 'STATUS_STOK', 'TERJUAL/BLN']], use_container_width=True)
        
        st.markdown("#### Produk dengan Harga Jual < HPP Terbaru (Potensi Kerugian)")
        st.dataframe(produk_lebih_murah[['NAMA', 'SKU', 'HARGA', 'HPP', 'STATUS_STOK', 'TERJUAL/BLN']], use_container_width=True)
    else:
        st.warning("Kolom 'HPP (LATEST)' tidak ditemukan di worksheet DATABASE.")

elif analysis_mode == "SIMILARITY PRODUK":
    st.header("🔍 Analisis Similaritas Produk (TF-IDF)")
    st.write("Pilih produk DB KLIK untuk menemukan produk serupa dari toko kompetitor berdasarkan data terbaru.")
    
    df_db_klik_sim = df_db_klik_latest.copy()
    df_kompetitor_sim = df_kompetitor_latest.copy()

    produk_list = sorted(df_db_klik_sim['NAMA'].unique())
    if produk_list:
        selected_product = st.selectbox("Pilih Produk DB KLIK:", produk_list)
        
        if selected_product:
            with st.spinner("Mencari produk serupa..."):
                matches = find_product_matches_tfidf(selected_product, df_db_klik_sim, df_kompetitor_sim)
                
                if not matches or len(matches) <= 1:
                    st.warning("Tidak ditemukan produk yang mirip di toko kompetitor.")
                    db_klik_product_info = df_db_klik_sim[df_db_klik_sim['NAMA'] == selected_product]
                    st.dataframe(db_klik_product_info[['NAMA', 'HARGA', 'STATUS_STOK']], use_container_width=True)
                else:
                    df_matches = pd.DataFrame(matches)
                    df_matches['OMZET'] = df_matches['HARGA'] * df_matches['TERJUAL/BLN']
                    
                    col11, col12, col13 = st.columns(3)
                    harga_kompetitor = df_matches[df_matches['TOKO'] != 'DB KLIK']['HARGA']
                    rata_rata_harga = harga_kompetitor.mean() if not harga_kompetitor.empty else 0
                    
                    col11.metric("Rata-rata Harga Kompetitor", f"Rp {rata_rata_harga:,.0f}")
                    col12.metric("Jumlah Toko Pesaing", f"{len(harga_kompetitor)}")
                    
                    omzet_tertinggi_row = df_matches[df_matches['TOKO'] != 'DB KLIK'].sort_values('OMZET', ascending=False)
                    if not omzet_tertinggi_row.empty:
                        omzet_tertinggi = omzet_tertinggi_row.iloc[0]
                        col13.metric("Pesaing Omzet Tertinggi", f"{omzet_tertinggi['TOKO']}", f"Rp {omzet_tertinggi['OMZET']:,.0f}")
                    else:
                        col13.metric("Pesaing Omzet Tertinggi", "-")

                    st.markdown("#### Tabel Perbandingan Produk")
                    display_cols = ['TOKO', 'NAMA', 'HARGA', 'STATUS_STOK', 'SKOR_KEMIRIPAN']
                    df_display_matches = df_matches[display_cols].sort_values('SKOR_KEMIRIPAN', ascending=False)
                    df_display_matches['HARGA'] = df_display_matches['HARGA'].apply(lambda x: f"Rp {x:,.0f}")
                    st.dataframe(df_display_matches, use_container_width=True)
    else:
        st.info("Tidak ada produk DB KLIK yang tersedia pada data tanggal terbaru.")

