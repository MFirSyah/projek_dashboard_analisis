# ===================================================================================
# IMPORT PUSTAKA YANG DIBUTUHKAN
# ===================================================================================
import streamlit as st # Untuk membuat aplikasi web interaktif
import pandas as pd # Untuk manipulasi dan analisis data (DataFrame)
import numpy as np # Untuk operasi numerik, terutama untuk mengganti nilai kosong
import plotly.express as px # Untuk membuat grafik interaktif yang canggih
import re # Untuk operasi Regular Expression (membersihkan teks)
from sklearn.feature_extraction.text import TfidfVectorizer # Untuk mengubah teks menjadi vektor TF-IDF
from sklearn.metrics.pairwise import cosine_similarity # Untuk menghitung kemiripan antar vektor
from io import BytesIO # Untuk menangani data biner di memori (untuk unduh file Excel)
import warnings # Untuk mengontrol pesan peringatan
import gspread # Untuk berinteraksi dengan Google Sheets API
from google.oauth2.service_account import Credentials # Untuk otentikasi dengan Google Cloud Platform

# Mengabaikan peringatan yang tidak relevan dari library openpyxl saat memproses file Excel
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ===================================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ===================================================================================
# st.set_page_config() harus menjadi perintah Streamlit pertama yang dijalankan.
# Ini mengatur properti global halaman seperti layout, judul, dan ikon.
st.set_page_config(
    layout="wide", # Menggunakan layout lebar agar konten memenuhi layar
    page_title="Mesin Analisis Kompetitor Cerdas", # Judul yang muncul di tab browser
    page_icon="🧠" # Ikon yang muncul di tab browser
)

# ===================================================================================
# FUNGSI-FUNGSI UTAMA (DATA LOADING, PROCESSING, ANALYSIS)
# ===================================================================================

def normalize_text(name):
    """
    Fungsi ini membersihkan dan menstandarkan nama produk.
    Tujuannya adalah agar perbandingan teks (TF-IDF) menjadi lebih akurat
    dengan menghilangkan karakter tidak penting dan kata-kata umum (stopwords).
    """
    if not isinstance(name, str): return "" # Jika input bukan string, kembalikan string kosong
    text = re.sub(r'[^\w\s.]', ' ', name.lower()) # Hapus semua karakter kecuali huruf, angka, spasi, dan titik
    # Standarkan satuan umum seperti inch, gb, tb, hz
    text = re.sub(r'(\d+)\s*inch|\"', r'\1 inch', text)
    text = re.sub(r'(\d+)\s*gb', r'\1gb', text)
    text = re.sub(r'(\d+)\s*tb', r'\1tb', text)
    text = re.sub(r'(\d+)\s*hz', r'\1hz', text)
    # Hapus kata-kata umum (stopwords) yang tidak memiliki nilai pembeda signifikan
    stopwords = [
        'garansi', 'resmi', 'original', 'dan', 'promo', 'murah', 'untuk', 'dengan', 
        'built', 'in', 'speaker', 'hdmi', 'vga', 'dp', 'type-c', 'usb', 'bluetooth',
        'wireless', 'gaming', 'keyboard', 'mouse', 'monitor', 'led', 'ips', 'va'
    ]
    tokens = [word for word in text.split() if word not in stopwords] # Pisahkan kalimat menjadi kata dan buang stopwords
    return ' '.join(tokens) # Gabungkan kembali kata-kata menjadi kalimat yang bersih

@st.cache_data(ttl=3600) # Decorator Streamlit untuk caching. Data akan disimpan selama 1 jam (3600 detik).
def load_and_process_data():
    """
    Fungsi inti untuk memuat data dari Google Sheets, membersihkan, menggabungkan,
    dan memprosesnya menjadi DataFrame yang siap dianalisis.
    """
    try:
        # --- LANGKAH 1: KONEKSI KE GOOGLE SHEETS ---
        # Menggunakan st.secrets untuk mengambil kredensial yang tersimpan aman di Streamlit
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"], # Lingkup akses: hanya baca
        )
        gc = gspread.authorize(creds) # Otorisasi koneksi
        
        # Buka spreadsheet menggunakan ID yang juga disimpan di secrets
        spreadsheet = gc.open_by_key(st.secrets["SOURCE_SPREADSHEET_ID"])
        
        # --- LANGKAH 2: MEMBACA SEMUA WORKSHEET MENJADI DATAFRAME ---
        worksheets = spreadsheet.worksheets() # Dapatkan daftar semua sheet
        data_frames = {} # Dictionary untuk menyimpan setiap sheet sebagai DataFrame
        for sheet in worksheets:
            data = sheet.get_all_values() # Ambil semua data dari sheet
            if len(data) > 1: # Pastikan sheet tidak kosong (minimal header + 1 baris data)
                headers = data.pop(0) # Ambil baris pertama sebagai header
                df = pd.DataFrame(data, columns=headers) # Buat DataFrame
                df.replace("", np.nan, inplace=True) # Ganti string kosong dengan NaN (Not a Number) agar mudah diproses
                data_frames[sheet.title] = df

        # --- LANGKAH 3: MEMUAT DATA REFERENSI (KAMUS & DATABASE) ---
        kamus_brand = data_frames['kamus_brand'].rename(columns={'Alias': 'NAMA', 'Brand_Utama': 'BRAND'})
        database_df = data_frames['DATABASE'].assign(NAMA_NORMALIZED=lambda df: df['NAMA'].apply(normalize_text))
        database_brand = data_frames['DATABASE_BRAND']['NAMA BRAND'].str.upper().tolist()

        # --- LANGKAH 4: MENGGABUNGKAN SEMUA DATA TOKO ---
        all_data_list = []
        toko_list = [ # Daftar toko yang akan dianalisis
            'DB KLIK', 'LOGITECH', 'ABDITAMA', 'LEVEL99', 'IT SHOP', 
            'JAYA PC', 'MULTIFUNGSI', 'TECH ISLAND', 'GG STORE', 'SURYA MITRA ONLINE'
        ]
        
        sheet_names = list(data_frames.keys())
        # Looping untuk setiap toko dan status (READY/HABIS) untuk membaca sheet yang relevan
        for toko in toko_list:
            for status in ['READY', 'HABIS']:
                # Mencari nama sheet yang cocok (misal: "DB KLIK - REKAP - READY")
                sheet_name_rekap = [s for s in sheet_names if toko in s and status in s and 'REKAP' in s]
                if sheet_name_rekap and sheet_name_rekap[0] in data_frames:
                    df = data_frames[sheet_name_rekap[0]].copy()
                    df['TOKO'] = toko # Tambahkan kolom 'TOKO'
                    df['STATUS'] = status # Tambahkan kolom 'STATUS'
                    all_data_list.append(df)

        # Gabungkan semua data dari list menjadi satu DataFrame besar
        df_gabungan = pd.concat(all_data_list, ignore_index=True)

        # --- LANGKAH 5: NORMALISASI & PEMBERSIHAN DATA GABUNGAN ---
        df_gabungan.rename(columns=lambda x: str(x).strip().upper(), inplace=True) # Jadikan semua nama kolom uppercase
        if 'TERJUAL/BLN' in df_gabungan.columns: # Ganti nama kolom agar konsisten
            df_gabungan.rename(columns={'TERJUAL/BLN': 'TERJUAL_PER_BLN'}, inplace=True)
        
        # Konversi kolom tanggal ke format datetime
        df_gabungan['TANGGAL'] = pd.to_datetime(df_gabungan['TANGGAL'], errors='coerce')
        # Buang baris yang tidak memiliki data penting (tanggal, nama, harga)
        df_gabungan = df_gabungan.dropna(subset=['TANGGAL', 'NAMA', 'HARGA'])
        
        # Konversi kolom numerik ke tipe data angka, ganti error dengan 0
        numeric_cols = ['HARGA', 'TERJUAL_PER_BLN']
        for col in numeric_cols:
            if col in df_gabungan.columns:
                df_gabungan[col] = pd.to_numeric(df_gabungan[col], errors='coerce').fillna(0)

        # Hitung omzet
        df_gabungan['OMZET'] = df_gabungan['HARGA'] * df_gabungan['TERJUAL_PER_BLN']

        # --- LANGKAH 6: NORMALISASI BRAND ---
        df_gabungan['BRAND'] = df_gabungan['BRAND'].astype(str).str.upper()
        kamus_brand_dict = dict(zip(kamus_brand['NAMA'].str.upper(), kamus_brand['BRAND'].str.upper()))
        # Gunakan kamus brand untuk mengganti alias dengan nama brand utama
        df_gabungan['BRAND'] = df_gabungan['BRAND'].replace(kamus_brand_dict)

        # Fungsi untuk mencoba mengekstrak brand dari nama produk jika kolom brand kosong
        def extract_brand(name):
            if not isinstance(name, str): return 'TIDAK DIKETAHUI'
            name_upper = name.upper()
            for brand in database_brand:
                if f' {brand} ' in f' {name_upper} ':
                    return brand
            return 'TIDAK DIKETAHUI'
        
        # Terapkan fungsi extract_brand
        df_gabungan['BRAND'] = df_gabungan.apply(
            lambda row: extract_brand(row['NAMA']) if pd.isna(row['BRAND']) or row['BRAND'] in ['TIDAK ADA BRAND', '', 'NAN'] else row['BRAND'],
            axis=1
        )
        
        return df_gabungan, database_df

    # --- BLOK PENANGANAN ERROR (EXCEPTION HANDLING) ---
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Spreadsheet tidak ditemukan. Periksa kembali `SOURCE_SPREADSHEET_ID` di secrets.toml.")
        return None, None
    except gspread.exceptions.APIError as e:
        st.error(f"Terjadi kesalahan API Google Sheets: {e}. Pastikan service account memiliki akses ke spreadsheet.")
        return None, None
    except KeyError as e:
        st.error(f"Gagal memproses data: Kolom atau sheet yang dibutuhkan tidak ditemukan: {e}. Periksa nama sheet dan kolom di Google Sheets Anda.")
        return None, None
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data dari Google Sheets: {e}")
        st.info("Pastikan konfigurasi `secrets.toml` sudah benar dan koneksi internet stabil.")
        return None, None

def perform_sku_labeling(df_db_klik, df_database):
    """
    Melakukan pelabelan SKU dan KATEGORI pada data DB KLIK secara otomatis
    menggunakan metode TF-IDF untuk mencari produk yang paling mirip di database.
    """
    # Tampilkan spinner saat proses berjalan
    with st.spinner("Melakukan pelabelan cerdas dengan TF-IDF... Ini mungkin memakan waktu beberapa saat."):
        # Filter hanya produk DB KLIK yang belum punya SKU atau KATEGORI
        df_to_label = df_db_klik[df_db_klik['KATEGORI'].isnull() | df_db_klik['SKU'].isnull()].copy()
        
        if df_to_label.empty:
            st.toast("Tidak ada produk baru yang perlu dilabeli.", icon="✅")
            return df_db_klik

        # Normalisasi nama produk yang akan dilabeli
        df_to_label['NAMA_NORMALIZED'] = df_to_label['NAMA'].apply(normalize_text)

        # Buat model TF-IDF berdasarkan nama produk di database
        vectorizer = TfidfVectorizer().fit(df_database['NAMA_NORMALIZED'])
        db_matrix = vectorizer.transform(df_database['NAMA_NORMALIZED'])
        
        labeled_indices = []
        labeled_data = []

        # Loop untuk setiap produk yang akan dilabeli
        for index, row in df_to_label.iterrows():
            # Ubah nama produk menjadi vektor TF-IDF
            product_matrix = vectorizer.transform([row['NAMA_NORMALIZED']])
            # Hitung kemiripan kosinus dengan semua produk di database
            cosine_sim = cosine_similarity(product_matrix, db_matrix)
            # Dapatkan indeks dari produk database yang paling mirip
            best_match_index = cosine_sim.argmax()
            
            # Ambil data SKU dan KATEGORI dari produk yang paling mirip
            matched_row = df_database.iloc[best_match_index]
            labeled_indices.append(index)
            labeled_data.append({'SKU': matched_row['SKU'], 'KATEGORI': matched_row['KATEGORI']})

        # Update DataFrame DB KLIK dengan data yang baru dilabeli
        if labeled_data:
            df_labels = pd.DataFrame(labeled_data, index=labeled_indices)
            df_db_klik.update(df_labels)

    st.toast(f"Pelabelan selesai untuk {len(df_to_label)} produk.", icon="✨")
    return df_db_klik

def find_matches_tfidf(selected_product_row, df_db_klik, df_kompetitor):
    """
    Mencari produk kompetitor yang mirip berdasarkan TF-IDF dengan validasi brand wajib.
    """
    brand = selected_product_row['BRAND']
    if brand == 'TIDAK DIKETAHUI':
        return []

    # LANGKAH 1: Filter kompetitor berdasarkan brand yang sama. Ini adalah aturan keras.
    df_competitor_filtered = df_kompetitor[(df_kompetitor['BRAND'] == brand) & (df_kompetitor['TOKO'] != 'DB KLIK')].copy()

    if df_competitor_filtered.empty:
        return []

    # LANGKAH 2: Lakukan perbandingan TF-IDF pada produk yang brand-nya sudah sama
    combined_df = pd.concat([pd.DataFrame([selected_product_row]), df_competitor_filtered], ignore_index=True)
    combined_df['NAMA_NORMALIZED'] = combined_df['NAMA'].apply(normalize_text)
    
    vectorizer = TfidfVectorizer().fit(combined_df['NAMA_NORMALIZED'])
    tfidf_matrix = vectorizer.transform(combined_df['NAMA_NORMALIZED'])
    
    # Hitung kemiripan dari produk kita (indeks 0) ke semua produk lain
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    
    matches = []
    # Loop mulai dari 1 untuk mengabaikan perbandingan dengan diri sendiri
    for i in range(1, len(combined_df)):
        score = cosine_sim[0, i]
        # Atur ambang batas kemiripan (misal: 40%)
        if score > 0.4:
            match_row = combined_df.iloc[i]
            selisih = match_row['HARGA'] - selected_product_row['HARGA']
            matches.append({
                'Nama Produk Tercantum': match_row['NAMA'],
                'Toko': match_row['TOKO'],
                'Harga': match_row['HARGA'],
                'Selisih Harga': selisih,
                'Status Stok': match_row['STATUS'],
                'Skor Kemiripan (%)': round(score * 100, 2)
            })
    return matches

# ===================================================================================
# UI (USER INTERFACE) STREAMLIT
# ===================================================================================
st.title("🧠 Mesin Analisis Kompetitor Cerdas")
st.markdown("Selamat datang, Firman! Platform ini dirancang untuk memberikan analisis mendalam terhadap data penjualan DB KLIK dan para kompetitornya.")

# --- Langkah 1: Memuat dan memproses data saat aplikasi pertama kali dijalankan ---
# st.session_state digunakan untuk menyimpan variabel antar-rerun aplikasi.
if 'data_loaded' not in st.session_state:
    with st.spinner("Menarik dan memproses data dari semua toko... Harap tunggu."):
        # Panggil fungsi pemuatan data
        df_gabungan, database_df = load_and_process_data()
        if df_gabungan is not None:
            # Simpan data ke session_state agar tidak perlu dimuat ulang setiap kali ada interaksi
            st.session_state.df_gabungan = df_gabungan
            st.session_state.database_df = database_df
            st.session_state.data_loaded = True
            st.session_state.needs_labeling_check = True
        else:
            st.stop() # Hentikan eksekusi jika data gagal dimuat

# --- Langkah 2: Cek apakah ada data yang perlu dilabeli ---
if st.session_state.get('needs_labeling_check', False):
    df_db_klik_latest = st.session_state.df_gabungan[
        (st.session_state.df_gabungan['TOKO'] == 'DB KLIK') & 
        (st.session_state.df_gabungan['TANGGAL'] == st.session_state.df_gabungan['TANGGAL'].max())
    ].copy()
    
    # Jika ada nilai kosong di kolom KATEGORI atau SKU, set status 'needs_labeling' menjadi True
    if df_db_klik_latest['KATEGORI'].isnull().any() or df_db_klik_latest['SKU'].isnull().any():
        st.session_state.needs_labeling = True
    else:
        st.session_state.needs_labeling = False
    st.session_state.needs_labeling_check = False # Cek hanya dilakukan sekali per pemuatan data

# --- Tampilkan Peringatan & Tombol Pelabelan jika diperlukan ---
if st.session_state.get('needs_labeling', False):
    st.warning(
        "**PELABELAN DIPERLUKAN!**\n\n"
        "Terdeteksi ada produk baru di DB KLIK pada tanggal terbaru yang belum memiliki data SKU dan Kategori. "
        "Silakan jalankan proses pelabelan untuk melanjutkan analisis."
    )
    if st.button("🚀 JALANKAN PELABELAN SKU DAN KATEGORI", type="primary"):
        df_db_klik = st.session_state.df_gabungan[st.session_state.df_gabungan['TOKO'] == 'DB KLIK'].copy()
        df_labeled = perform_sku_labeling(df_db_klik, st.session_state.database_df)
        
        # Update dataframe utama di session_state dengan hasil pelabelan
        st.session_state.df_gabungan.update(df_labeled)
        st.session_state.needs_labeling = False
        st.rerun() # Muat ulang aplikasi untuk menampilkan konten utama
    st.stop() # Hentikan eksekusi sampai tombol pelabelan diklik

# --- Jika data sudah siap dan terlabel, tampilkan aplikasi utama ---
df_gabungan = st.session_state.df_gabungan
database_df = st.session_state.database_df

# ===================================================================================
# SIDEBAR - Panel Navigasi dan Kontrol
# ===================================================================================
with st.sidebar:
    st.header("⚙️ Panel Kontrol")

    # --- Widget Pemilihan Rentang Tanggal ---
    min_date = df_gabungan['TANGGAL'].min().date()
    max_date = df_gabungan['TANGGAL'].max().date()
    date_range = st.date_input(
        "Pilih Rentang Tanggal Analisis",
        (min_date, max_date), # Nilai default: dari tanggal terlama hingga terbaru
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
    )
    
    if len(date_range) != 2:
        st.stop() # Pastikan user memilih rentang tanggal yang valid
    
    start_date, end_date = date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # --- Filter data utama berdasarkan tanggal yang dipilih user ---
    df_filtered = df_gabungan[
        (df_gabungan['TANGGAL'] >= start_date) & (df_gabungan['TANGGAL'] <= end_date)
    ].copy()
    
    # Buat juga DataFrame khusus untuk data tanggal terbaru saja
    df_latest = df_gabungan[df_gabungan['TANGGAL'] == df_gabungan['TANGGAL'].max()].copy()

    # --- Informasi Data Sederhana ---
    st.info(f"Data dari **{min_date}** hingga **{max_date}**.")
    st.metric("Total Baris Data Dianalisis", f"{len(df_filtered):,}")

    # --- Navigasi Halaman Analisis ---
    page = st.radio(
        "Pilih Jenis Analisis",
        ["📊 ANALISIS UTAMA", "💰 HPP PRODUK", "🔗 SIMILARITY PRODUK"]
    )

    # --- Tombol untuk Menjalankan Ulang Pelabelan ---
    st.markdown("---")
    if st.button("Jalankan Ulang Pelabelan SKU & Kategori"):
        df_db_klik = df_gabungan[df_gabungan['TOKO'] == 'DB KLIK'].copy()
        df_labeled = perform_sku_labeling(df_db_klik, database_df)
        st.session_state.df_gabungan.update(df_labeled)
        st.rerun()
        
    # --- Tombol untuk Mengunduh Data ---
    @st.cache_data # Cache hasil konversi ke Excel agar tidak perlu diproses ulang
    def to_excel(df):
        output = BytesIO() # Buat file di memori
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Hasil Analisis')
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df_filtered)
    st.download_button(
        label="📥 Unduh Data Excel (Terfilter)",
        data=excel_data,
        file_name=f"analisis_data_{start_date.date()}_to_{end_date.date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===================================================================================
# KONTEN UTAMA - Tampilan berdasarkan pilihan di sidebar
# ===================================================================================

# --- Fungsi Bantuan untuk Formatting Angka menjadi Rupiah ---
def format_rupiah(angka):
    return f"Rp {angka:,.0f}".replace(",",".")

# --- KONTEN HALAMAN: ANALISIS UTAMA ---
if page == "📊 ANALISIS UTAMA":
    st.header("📊 Analisis Utama")
    
    # Buat sistem tab untuk memisahkan analisis
    tab1, tab2, tab3 = st.tabs([
        "Analisis DB KLIK", 
        "Analisis Kompetitor", 
        "Perbandingan Produk Baru & Habis"
    ])

    # --- KONTEN TAB 1: ANALISIS DB KLIK ---
    with tab1:
        st.subheader("📈 Analisis Performa DB KLIK")
        # Filter data hanya untuk DB KLIK
        df_db_klik_filtered = df_filtered[df_filtered['TOKO'] == 'DB KLIK'].copy()
        df_db_klik_latest = df_latest[df_latest['TOKO'] == 'DB KLIK'].copy()

        # Tampilkan metrik utama
        col1, col2 = st.columns([1,2])
        with col1:
             st.metric("Total Omzet (Rentang Waktu)", format_rupiah(df_db_klik_filtered['OMZET'].sum()))
             st.metric("Total Unit Terjual (Rentang Waktu)", f"{int(df_db_klik_filtered['TERJUAL_PER_BLN'].sum()):,}")
        with col2:
            st.info("💡 **Tips:** Gunakan panel kontrol di sebelah kiri untuk mengubah rentang tanggal analisis.")

        st.markdown("---")

        # Analisis Peringkat Kategori
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Peringkat Kategori Berdasarkan Omzet")
            cat_omzet = df_db_klik_filtered.groupby('KATEGORI')['OMZET'].sum().sort_values(ascending=False).reset_index()
            
            # Kontrol interaktif untuk user
            num_bars = st.slider("Jumlah Kategori Ditampilkan:", 5, 50, 10)
            sort_order = st.radio("Urutkan:", ('Tertinggi ke Terendah', 'Terendah ke Tertinggi'), horizontal=True)
            ascending = sort_order == 'Terendah ke Tertinggi'
            
            cat_omzet_sorted = cat_omzet.sort_values('OMZET', ascending=ascending).tail(num_bars)
            
            # Buat bar chart dengan Plotly
            fig = px.bar(cat_omzet_sorted, x='OMZET', y='KATEGORI', orientation='h', 
                         title=f'Top {num_bars} Kategori', text_auto='.2s')
            fig.update_layout(yaxis={'categoryorder':'total ascending' if ascending else 'total descending'})
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Omzet per Kategori (Tabel)")
            cat_omzet['OMZET'] = cat_omzet['OMZET'].apply(format_rupiah)
            st.dataframe(cat_omzet, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Produk Teratas")
        
        # Ambil nama kategori teratas dari hasil sorting sebelumnya
        top_cat = cat_omzet_sorted['KATEGORI'].iloc[-1]
        
        # Tampilkan produk dalam kategori teratas
        st.markdown(f"**Produk dalam Kategori Teratas: __{top_cat}__ (berdasarkan data terbaru)**")
        df_top_cat_prod = df_db_klik_latest[df_db_klik_latest['KATEGORI'] == top_cat][['NAMA', 'HARGA', 'TERJUAL_PER_BLN', 'OMZET']].sort_values('OMZET', ascending=False).head(10)
        df_top_cat_prod['HARGA'] = df_top_cat_prod['HARGA'].apply(format_rupiah)
        df_top_cat_prod['OMZET'] = df_top_cat_prod['OMZET'].apply(format_rupiah)
        st.dataframe(df_top_cat_prod, use_container_width=True)
        
        # Tampilkan produk terlaris secara keseluruhan
        st.markdown("**Produk Terlaris Global (berdasarkan rentang waktu)**")
        df_top_global = df_db_klik_filtered[['NAMA', 'SKU', 'HARGA', 'OMZET']].sort_values('OMZET', ascending=False).head(10)
        df_top_global['HARGA'] = df_top_global['HARGA'].apply(format_rupiah)
        df_top_global['OMZET'] = df_top_global['OMZET'].apply(format_rupiah)
        st.dataframe(df_top_global, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Analisis Brand DB KLIK")
        c1, c2 = st.columns(2)
        with c1:
            # Buat pie chart untuk brand teratas berdasarkan data terbaru
            brand_omzet_latest = df_db_klik_latest.groupby('BRAND')['OMZET'].sum().nlargest(6).reset_index()
            fig = px.pie(brand_omzet_latest, values='OMZET', names='BRAND', 
                         title='Top 6 Brand Berdasarkan Omzet (Data Terbaru)',
                         hole=.3, hover_data={'OMZET':':,.0f'})
            fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Tampilkan tabel ringkasan brand berdasarkan rentang waktu yang dipilih
            brand_summary = df_db_klik_filtered.groupby('BRAND').agg(
                Total_Omzet=('OMZET', 'sum'),
                Total_Unit_Terjual=('TERJUAL_PER_BLN', 'sum')
            ).sort_values('Total_Omzet', ascending=False).reset_index()
            
            brand_summary['Total_Omzet'] = brand_summary['Total_Omzet'].apply(format_rupiah)
            st.dataframe(brand_summary, use_container_width=True, height=350)

    # --- KONTEN TAB 2: ANALISIS KOMPETITOR ---
    with tab2:
        st.subheader("⚔️ Analisis Kompetitor")
        
        # Bar chart perbandingan omzet total per toko
        omzet_per_toko = df_filtered.groupby('TOKO')['OMZET'].sum().sort_values(ascending=False).reset_index()
        fig_omzet_toko = px.bar(omzet_per_toko, x='TOKO', y='OMZET', title='Perbandingan Omzet Total per Toko', text_auto='.2s')
        st.plotly_chart(fig_omzet_toko, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Tren Pendapatan per Toko")
        
        # Line chart untuk melihat tren omzet harian
        line_data = df_filtered.groupby(['TANGGAL', 'TOKO'])['OMZET'].sum().reset_index()
        fig_line = px.line(line_data, x='TANGGAL', y='OMZET', color='TOKO', title='Tren Omzet Harian per Toko')
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Expander untuk menampilkan data mentah dari chart
        with st.expander("Lihat Data Tabel Tren Pendapatan"):
            pivot_table = line_data.pivot(index='TANGGAL', columns='TOKO', values='OMZET').fillna(0)
            st.dataframe(pivot_table.style.format(format_rupiah), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Tren Produk Ready vs Habis per Toko")
        
        # Line chart untuk membandingkan jumlah produk ready vs habis
        status_data = df_filtered.groupby(['TANGGAL', 'TOKO', 'STATUS']).size().reset_index(name='JUMLAH_PRODUK')
        fig_status = px.line(status_data, x='TANGGAL', y='JUMLAH_PRODUK', color='TOKO', facet_row='STATUS',
                             title='Jumlah Produk Ready vs Habis dari Waktu ke Waktu', height=600)
        st.plotly_chart(fig_status, use_container_width=True)
        
        with st.expander("Lihat Data Tabel Produk Ready vs Habis"):
             st.dataframe(status_data, use_container_width=True)
             
    # --- KONTEN TAB 3: PERBANDINGAN PRODUK BARU & HABIS ---
    with tab3:
        st.subheader("🔄 Perbandingan Produk Baru dan Habis")
        st.info("Fitur ini membandingkan daftar produk unik antara dua tanggal yang dipilih untuk setiap toko.")
        
        all_dates = sorted(df_gabungan['TANGGAL'].dt.date.unique(), reverse=True)
        
        # Input untuk memilih tanggal pembanding dan tanggal target
        c1, c2 = st.columns(2)
        with c1:
            date_compare = st.selectbox("Pilih Tanggal Pembanding:", all_dates)
        with c2:
            date_target = st.selectbox("Pilih Tanggal Target:", all_dates, index=0)
            
        if date_compare and date_target:
            df_compare = df_gabungan[df_gabungan['TANGGAL'].dt.date == date_compare]
            df_target = df_gabungan[df_gabungan['TANGGAL'].dt.date == date_target]

            # Loop untuk setiap toko, tampilkan perbandingan dalam expander
            for toko in df_gabungan['TOKO'].unique():
                with st.expander(f"Analisis Toko: **{toko}**"):
                    # Gunakan 'set' untuk mencari perbedaan produk dengan cepat
                    set_compare = set(df_compare[df_compare['TOKO'] == toko]['NAMA'])
                    set_target = set(df_target[df_target['TOKO'] == toko]['NAMA'])
                    
                    produk_baru = set_target - set_compare
                    produk_hilang = set_compare - set_target
                    
                    # Tampilkan hasilnya dalam dua kolom
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### Produk Baru/Muncul Kembali ({len(produk_baru)})")
                        if produk_baru:
                            st.dataframe(pd.DataFrame(list(produk_baru), columns=['Nama Produk']), use_container_width=True)
                        else:
                            st.write("Tidak ada produk baru.")
                    
                    with col2:
                        st.markdown(f"#### Produk Hilang/Habis ({len(produk_hilang)})")
                        if produk_hilang:
                            st.dataframe(pd.DataFrame(list(produk_hilang), columns=['Nama Produk']), use_container_width=True)
                        else:
                            st.write("Tidak ada produk yang hilang.")

# --- KONTEN HALAMAN: HPP PRODUK ---
elif page == "💰 HPP PRODUK":
    st.header("💰 Analisis Harga Pokok Penjualan (HPP)")
    st.info("Membandingkan harga jual produk DB KLIK (Ready & Habis) terbaru dengan HPP dari worksheet DATABASE.")

    df_db_klik_combined = df_gabungan[df_gabungan['TOKO'] == 'DB KLIK'].copy()
    
    # Ambil harga terbaru untuk setiap SKU dengan mengurutkan berdasarkan tanggal dan menghapus duplikat
    df_db_klik_latest_price = df_db_klik_combined.sort_values('TANGGAL').drop_duplicates('SKU', keep='last')
    
    # Gabungkan (merge) data harga terbaru dengan data HPP dari database berdasarkan SKU
    df_compare_hpp = pd.merge(
        df_db_klik_latest_price[['NAMA', 'SKU', 'HARGA', 'STATUS', 'TERJUAL_PER_BLN']],
        database_df[['SKU', 'HPP (LATEST)']],
        on='SKU',
        how='inner' # Hanya ambil produk yang ada di kedua tabel
    )
    df_compare_hpp.rename(columns={'HPP (LATEST)': 'HPP'}, inplace=True)
    df_compare_hpp = df_compare_hpp.dropna(subset=['HPP']) # Hapus jika tidak ada data HPP
    df_compare_hpp['SELISIH_HPP'] = df_compare_hpp['HARGA'] - df_compare_hpp['HPP']
    
    # Pisahkan produk yang dijual lebih mahal dan lebih murah dari HPP
    df_lebih_mahal = df_compare_hpp[df_compare_hpp['SELISIH_HPP'] > 0].sort_values('SELISIH_HPP', ascending=False)
    df_lebih_murah = df_compare_hpp[df_compare_hpp['SELISIH_HPP'] < 0].sort_values('SELISIH_HPP', ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🟢 Produk Lebih Mahal dari HPP ({len(df_lebih_mahal)})")
        df_display = df_lebih_mahal[['NAMA', 'SKU', 'HARGA', 'HPP', 'STATUS', 'TERJUAL_PER_BLN']]
        df_display['HARGA'] = df_display['HARGA'].apply(format_rupiah)
        df_display['HPP'] = df_display['HPP'].apply(format_rupiah)
        st.dataframe(df_display, use_container_width=True, height=600)
        
    with col2:
        st.subheader(f"🔴 Produk Lebih Murah dari HPP ({len(df_lebih_murah)})")
        df_display = df_lebih_murah[['NAMA', 'SKU', 'HARGA', 'HPP', 'STATUS', 'TERJUAL_PER_BLN']]
        df_display['HARGA'] = df_display['HARGA'].apply(format_rupiah)
        df_display['HPP'] = df_display['HPP'].apply(format_rupiah)
        st.dataframe(df_display, use_container_width=True, height=600)

# --- KONTEN HALAMAN: SIMILARITY PRODUK ---
elif page == "🔗 SIMILARITY PRODUK":
    st.header("🔗 Analisis Kemiripan Produk (TF-IDF)")
    st.info("Pilih produk DB KLIK dari data terbaru untuk menemukan produk serupa di toko kompetitor.")

    df_db_klik_latest = df_latest[df_latest['TOKO'] == 'DB KLIK'].copy()
    
    # Buat dropdown (selectbox) untuk memilih produk
    product_list = sorted(df_db_klik_latest['NAMA'].unique())
    selected_product_name = st.selectbox(
        "Pilih Produk DB KLIK untuk Dianalisis:",
        product_list,
        index=None, # Tidak ada yang dipilih secara default
        placeholder="Ketik untuk mencari produk..."
    )

    # Jika user sudah memilih produk, jalankan analisis
    if selected_product_name:
        selected_product_row = df_db_klik_latest[df_db_klik_latest['NAMA'] == selected_product_name].iloc[0]
        
        # Tampilkan detail produk yang dipilih
        st.markdown("---")
        st.subheader("Produk Referensi (DB KLIK)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Harga", format_rupiah(selected_product_row['HARGA']))
        c2.metric("Brand", selected_product_row['BRAND'])
        c3.metric("SKU", selected_product_row['SKU'])

        # Panggil fungsi pencarian kemiripan
        with st.spinner(f"Mencari produk yang mirip dengan '{selected_product_name}'..."):
            matches = find_matches_tfidf(selected_product_row, df_db_klik_latest, df_latest)

        st.markdown("---")
        st.subheader("Hasil Perbandingan di Toko Kompetitor")
        
        # Jika ditemukan produk yang mirip
        if matches:
            df_matches = pd.DataFrame(matches)
            
            # Tampilkan ringkasan metrik dari hasil pencarian
            c1, c2, c3 = st.columns(3)
            c1.metric("Rata-rata Harga Kompetitor", format_rupiah(df_matches['Harga'].mean()))
            c2.metric("Jumlah Toko Kompetitor", f"{df_matches['Toko'].nunique()} Toko")
            
            # Cari toko dengan omzet tertinggi dari produk yang mirip
            top_omzet_store_info = df_latest.loc[df_matches.index].groupby('TOKO')['OMZET'].sum().idxmax()
            top_omzet_value = df_latest.loc[df_matches.index].groupby('TOKO')['OMZET'].sum().max()
            c3.metric("Toko Omzet Tertinggi", f"{top_omzet_store_info} ({format_rupiah(top_omzet_value)})")
            
            # Tampilkan tabel hasil pencarian
            df_matches['Harga'] = df_matches['Harga'].apply(format_rupiah)
            df_matches['Selisih Harga'] = df_matches['Selisih Harga'].apply(format_rupiah)
            st.dataframe(df_matches.sort_values('Skor Kemiripan (%)', ascending=False), use_container_width=True)

        else:
            # Tampilkan pesan jika tidak ada produk yang mirip ditemukan
            st.warning("Tidak ditemukan produk yang mirip di toko kompetitor berdasarkan kriteria saat ini (Brand sama & skor kemiripan > 40%).")

