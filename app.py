# ===================================================================================
#  DASHBOARD ANALISIS PENJUALAN & KOMPETITOR - VERSI 6.0 (TF-IDF + BRAND VALIDATION)
#  Dibuat oleh: Firman & Asisten AI Gemini
#  Versi ini mengimplementasikan mesin pencocokan produk yang cerdas menggunakan
#  TF-IDF dan validasi brand 100% serta memindahkan ID Spreadsheet ke Secrets.
# ===================================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import re
import gspread
from datetime import datetime
from gspread_dataframe import set_with_dataframe
import numpy as np
# --- BARU: Import library untuk analisis TF-IDF ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Penjualan dan Kompetitor")

# ================================
# FUNGSI KONEKSI & NORMALISASI (MESIN BARU)
# ================================
@st.cache_resource(show_spinner="Menghubungkan ke Google Sheets...")
def connect_to_gsheets():
    creds_dict = {
        "type": st.secrets["gcp_type"], "project_id": st.secrets["gcp_project_id"],
        "private_key_id": st.secrets["gcp_private_key_id"], "private_key": st.secrets["gcp_private_key_raw"].replace('\\n', '\n'),
        "client_email": st.secrets["gcp_client_email"], "client_id": st.secrets["gcp_client_id"],
        "auth_uri": st.secrets["gcp_auth_uri"], "token_uri": st.secrets["gcp_token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_client_x509_cert_url"]
    }
    gc = gspread.service_account_from_dict(creds_dict)
    return gc

def normalize_text(name):
    """Membersihkan dan menstandarkan nama produk."""
    if not isinstance(name, str): return ""
    text = re.sub(r'[^\w\s.]', ' ', name.lower())
    text = re.sub(r'(\d+)\s*inch|\"', r'\1 inch', text)
    text = re.sub(r'(\d+)\s*gb', r'\1gb', text)
    text = re.sub(r'(\d+)\s*tb', r'\1tb', text)
    text = re.sub(r'(\d+)\s*hz', r'\1hz', text)
    text = text.replace('-', ' ')
    tokens = text.split()
    stopwords = [
        'garansi', 'resmi', 'original', 'dan', 'promo', 'murah', 'untuk',
        'dengan', 'built', 'in', 'speaker', 'hdmi', 'dp', 'vga', 'ms', 'office'
    ]
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

@st.cache_data(show_spinner="Memuat kamus brand...")
def load_brand_kamus(_gc, spreadsheet_key):
    """Memuat kamus brand dari Google Sheets untuk normalisasi."""
    try:
        spreadsheet = _gc.open_by_key(spreadsheet_key)
        kamus_sheet = spreadsheet.worksheet("kamus_brand")
        kamus_df = pd.DataFrame(kamus_sheet.get_all_records())
        if 'Alias' in kamus_df.columns and 'Brand_Utama' in kamus_df.columns:
            brand_map = dict(zip(kamus_df['Alias'].str.upper(), kamus_df['Brand_Utama'].str.upper()))
            return brand_map
    except Exception as e:
        st.warning(f"Gagal memuat 'kamus_brand': {e}. Normalisasi brand tidak akan seakurat seharusnya.")
    return {}

# ================================
# FUNGSI MEMUAT SEMUA DATA (DIPERBARUI)
# ================================
@st.cache_data(ttl=600, show_spinner="Mengambil data terbaru dari Google Sheets...")
def load_all_data(spreadsheet_key):
    gc = connect_to_gsheets()
    brand_map = load_brand_kamus(gc, spreadsheet_key)

    def normalize_brand(brand):
        if not isinstance(brand, str): return "LAINNYA"
        brand_upper = brand.upper()
        return brand_map.get(brand_upper, brand_upper)

    try:
        spreadsheet = gc.open_by_key(spreadsheet_key)
    except Exception as e:
        st.error(f"GAGAL KONEKSI/OPEN SPREADSHEET: {e}"); return None, None, None

    all_sheets = spreadsheet.worksheets()
    rekap_list_df, database_df = [], pd.DataFrame()

    for worksheet in all_sheets:
        sheet_name = worksheet.title
        try:
            if "DATABASE" == sheet_name.upper():
                database_df = pd.DataFrame(worksheet.get_all_records())
            elif "REKAP" in sheet_name.upper():
                all_values = worksheet.get_all_values()
                if not all_values or len(all_values) < 2: continue
                header, data = all_values[0], all_values[1:]
                df_sheet = pd.DataFrame(data, columns=header)
                if '' in df_sheet.columns: df_sheet = df_sheet.drop(columns=[''])
                
                store_name_match = re.match(r"^(.*?) - REKAP", sheet_name, re.IGNORECASE)
                toko_name = store_name_match.group(1).strip() if store_name_match else "Toko Tak Dikenal"
                df_sheet['Toko'] = toko_name
                df_sheet['Status'] = 'Tersedia' if "READY" in sheet_name.upper() else 'Habis'
                rekap_list_df.append(df_sheet)
        except Exception as e:
            st.warning(f"Gagal memproses sheet '{sheet_name}': {e}")
            continue

    if not rekap_list_df:
        st.error("Tidak ada data REKAP yang berhasil dimuat."); return None, None, None

    rekap_df = pd.concat(rekap_list_df, ignore_index=True)
    rekap_df.columns = [str(c).strip().upper() for c in rekap_df.columns]
    
    final_rename = {
        'NAMA': 'Nama Produk', 'TERJUAL/BLN': 'Terjual per Bulan', 'TANGGAL': 'Tanggal', 
        'HARGA': 'Harga', 'BRAND': 'Brand', 'STOK': 'Stok', 'TOKO': 'Toko', 
        'STATUS': 'Status', 'KATEGORI': 'Kategori', 'SKU': 'SKU'
    }
    rekap_df.rename(columns=final_rename, inplace=True)

    for col in ['Nama Produk', 'Brand', 'Toko', 'Status', 'Kategori', 'SKU']:
        if col in rekap_df.columns: rekap_df[col] = rekap_df[col].astype(str).str.strip()
    
    if 'Tanggal' in rekap_df.columns: rekap_df['Tanggal'] = pd.to_datetime(rekap_df['Tanggal'], errors='coerce', dayfirst=True)
    if 'Harga' in rekap_df.columns: rekap_df['Harga'] = pd.to_numeric(rekap_df['Harga'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')
    if 'Terjual per Bulan' in rekap_df.columns: rekap_df['Terjual per Bulan'] = pd.to_numeric(rekap_df['Terjual per Bulan'], errors='coerce').fillna(0)

    rekap_df.dropna(subset=['Tanggal', 'Nama Produk', 'Harga', 'Toko'], inplace=True)
    
    if 'Brand' not in rekap_df.columns: rekap_df['Brand'] = rekap_df['Nama Produk'].str.split().str[0]
    rekap_df['Brand'].fillna("LAINNYA", inplace=True)
    rekap_df['Brand'] = rekap_df['Brand'].apply(normalize_brand)
    
    rekap_df['Omzet'] = (rekap_df['Harga'].fillna(0) * rekap_df.get('Terjual per Bulan', 0).fillna(0)).astype(int)

    matches_df = pd.DataFrame()
    try:
        matches_sheet = spreadsheet.worksheet("HASIL_MATCHING")
        matches_df = pd.DataFrame(matches_sheet.get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        st.info("Worksheet 'HASIL_MATCHING' tidak ditemukan. Perlu dijalankan pembaruan.")
    except Exception as e:
        st.warning(f"Gagal memuat 'HASIL_MATCHING': {e}")

    return rekap_df.sort_values('Tanggal'), database_df, matches_df

# ================================
# FUNGSI SMART COMPARISON (MESIN BARU)
# ================================
def run_smart_comparison_update(gc, spreadsheet_key, score_cutoff=0.6):
    placeholder = st.empty()
    with placeholder.container():
        st.info("Memulai pembaruan perbandingan cerdas...")
        prog = st.progress(0, text="Langkah 1/4: Memuat data sumber...")

    load_all_data.clear() # Hapus cache agar data benar-benar baru
    df_full, _, _ = load_all_data(spreadsheet_key)
    if df_full is None or df_full.empty:
        with placeholder.container(): st.error("Gagal memuat data sumber. Batal."); return
    
    source_df = df_full.loc[df_full.groupby(['Toko', 'Nama Produk'])['Tanggal'].idxmax()]
    my_store_df = source_df[source_df['Toko'] == "DB KLIK"].copy()
    competitor_df = source_df[source_df['Toko'] != "DB KLIK"].copy()
    if my_store_df.empty or competitor_df.empty:
        with placeholder.container(): st.warning("Data toko Anda atau kompetitor tidak cukup."); return

    prog.progress(25, text="Langkah 2/4: Normalisasi nama produk...")
    my_store_df['Nama Normalisasi'] = my_store_df['Nama Produk'].apply(normalize_text)
    competitor_df['Nama Normalisasi'] = competitor_df['Nama Produk'].apply(normalize_text)

    all_matches = []
    total = len(my_store_df)
    prog.progress(50, text=f"Langkah 3/4: Mencocokkan produk 0/{total}...")
    
    all_normalized_names = pd.concat([my_store_df['Nama Normalisasi'], competitor_df['Nama Normalisasi']]).unique()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=1)
    vectorizer.fit(all_normalized_names)

    for i, (_, row) in enumerate(my_store_df.iterrows()):
        prog.progress(50 + int((i / total) * 40), text=f"Langkah 3/4: Mencocokkan produk {i+1}/{total}")
        selected_brand = row['Brand']
        
        competitor_brand_filtered = competitor_df[competitor_df['Brand'] == selected_brand]
        if competitor_brand_filtered.empty: continue

        vector_selected = vectorizer.transform([row['Nama Normalisasi']])
        vectors_competitor = vectorizer.transform(competitor_brand_filtered['Nama Normalisasi'])
        similarities = cosine_similarity(vector_selected, vectors_competitor)[0]
        
        for j, score in enumerate(similarities):
            if score >= score_cutoff:
                match = competitor_brand_filtered.iloc[j]
                all_matches.append({
                    'ID Produk Master (SKU)': row.get('SKU', 'N/A'),
                    'Nama Produk Master': row['Nama Produk'], 'Harga Master': int(row['Harga']),
                    'Produk Kompetitor': match['Nama Produk'], 'Harga Kompetitor': int(match['Harga']),
                    'Toko Kompetitor': match['Toko'], 'Status Stok Kompetitor': match['Status'],
                    'Skor Kemiripan': int(score * 100), 'Tanggal Update': datetime.now().strftime('%Y-%m-%d')
                })

    prog.progress(95, text="Langkah 4/4: Menyimpan hasil ke Google Sheets...")
    try:
        spreadsheet = gc.open_by_key(spreadsheet_key)
        try:
            worksheet = spreadsheet.worksheet("HASIL_MATCHING")
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="HASIL_MATCHING", rows=1, cols=1)
        
        if all_matches:
            results_df = pd.DataFrame(all_matches)
            set_with_dataframe(worksheet, results_df, resize=True)
            with placeholder.container(): st.success(f"Pembaruan Selesai: {len(results_df)} baris hasil perbandingan disimpan.")
        else:
            with placeholder.container(): st.warning("Tidak ditemukan pasangan produk yang cocok.")
    except Exception as e:
        with placeholder.container(): st.error(f"Gagal menyimpan hasil: {e}")

# ================================
# FUNGSI-FUNGSI PEMBANTU (UTILITY)
# ================================
def format_rupiah(val):
    if pd.isna(val) or not isinstance(val, (int, float, np.number)): return "N/A"
    return f"Rp {int(val):,}"
    
def format_wow_growth(pct_change):
    if pd.isna(pct_change) or pct_change == float('inf'): return "N/A"
    elif pct_change > 0.001: return f"▲ {pct_change:.1%}"
    elif pct_change < -0.001: return f"▼ {pct_change:.1%}"
    else: return f"▬ 0.0%"

def style_wow_growth(val):
    color = 'green' if '▲' in str(val) else 'red' if '▼' in str(val) else 'black'
    return f'color: {color}'

@st.cache_data
def convert_df_for_download(df):
    return df.to_csv(index=False).encode('utf-8')

# ================================
# APLIKASI UTAMA (MAIN APP)
# ================================
st.title("📊 Dashboard Analisis Penjualan & Bisnis v6.0")

try:
    SPREADSHEET_KEY = st.secrets["SOURCE_SPREADSHEET_ID"]
except KeyError:
    st.error("ID Spreadsheet belum diatur di Secrets. Mohon atur `SOURCE_SPREADSHEET_ID`."); st.stop()

gc = connect_to_gsheets()

if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if not st.session_state.data_loaded:
    if st.button("Tarik Data & Mulai Analisis 🚀", type="primary"):
        df_data, db_df_data, matches_df_data = load_all_data(SPREADSHEET_KEY)
        if df_data is not None and not df_data.empty:
            st.session_state.df, st.session_state.db_df, st.session_state.matches_df = df_data, db_df_data, matches_df_data
            st.session_state.data_loaded = True
            st.rerun()
        else: st.error("Gagal memuat data. Periksa akses Google Sheets.")
    st.info("👆 Klik tombol untuk menarik semua data yang diperlukan untuk analisis.")
    st.stop()

df = st.session_state.df
db_df = st.session_state.get('db_df', pd.DataFrame())
matches_df = st.session_state.get('matches_df', pd.DataFrame())

# ================================
# SIDEBAR (KONTROL UTAMA)
# ================================
st.sidebar.header("Mode Tampilan")
app_mode = st.sidebar.radio("Pilih Tampilan:", ("Tab Analisis", "HPP Produk"))
st.sidebar.divider()

if app_mode == "Tab Analisis":
    st.sidebar.header("Kontrol Analisis Umum")
    min_date_val, max_date_val = df['Tanggal'].min().date(), df['Tanggal'].max().date()
    start_date, end_date = st.sidebar.date_input("Rentang Tanggal:", [min_date_val, max_date_val], min_value=min_date_val, max_value=max_date_val)
    if len((start_date, end_date)) != 2: st.sidebar.warning("Pilih 2 tanggal."); st.stop()
    
    st.sidebar.divider()
    st.sidebar.header("Kontrol Perbandingan Produk")
    accuracy_cutoff = st.sidebar.slider("Tingkat Akurasi Minimum (%)", 50, 100, 65, 1)

    latest_source_date = df['Tanggal'].max().date()
    last_destination_update = datetime(1970, 1, 1).date()
    if not matches_df.empty and 'Tanggal Update' in matches_df.columns:
        matches_df['Tanggal Update'] = pd.to_datetime(matches_df['Tanggal Update'], errors='coerce')
        if not matches_df['Tanggal Update'].isna().all(): last_destination_update = matches_df['Tanggal Update'].max().date()

    st.sidebar.info(f"Data Sumber: **{latest_source_date.strftime('%d %b %Y')}**")
    st.sidebar.info(f"Perbandingan: **{last_destination_update.strftime('%d %b %Y')}**")
    
    if latest_source_date > last_destination_update:
        st.sidebar.warning("Data sumber lebih baru.")
        if st.sidebar.button("Perbarui Perbandingan Sekarang 🚀", type="primary"):
            run_smart_comparison_update(gc, SPREADSHEET_KEY, score_cutoff=accuracy_cutoff/100)
            load_all_data.clear() # Hapus cache agar data baru bisa diambil
            _, _, st.session_state.matches_df = load_all_data(SPREADSHEET_KEY)
            st.success("Pembaruan selesai!"); st.rerun()
    else:
        st.sidebar.success("Data perbandingan sudah terbaru.")
        
    if st.sidebar.button("Jalankan Pembaruan Manual"):
        run_smart_comparison_update(gc, SPREADSHEET_KEY, score_cutoff=accuracy_cutoff/100)
        load_all_data.clear() # Hapus cache agar data baru bisa diambil
        _, _, st.session_state.matches_df = load_all_data(SPREADSHEET_KEY)
        st.success("Pembaruan manual selesai!"); st.rerun()
        
    st.sidebar.divider()
    df_filtered_export = df[(df['Tanggal'].dt.date >= pd.to_datetime(start_date).date()) & (df['Tanggal'].dt.date <= pd.to_datetime(end_date).date())]
    st.sidebar.header("Ekspor & Info")
    st.sidebar.info(f"Baris data dalam rentang: **{len(df_filtered_export)}**")
    csv_data = convert_df_for_download(df_filtered_export)
    st.sidebar.download_button("📥 Unduh CSV (Filter)", data=csv_data, file_name=f'analisis_{start_date}_{end_date}.csv', mime='text/csv')

# ================================
# PERSIAPAN DATA UNTUK TABS
# ================================
start_date_dt, end_date_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
df_filtered = df[(df['Tanggal'] >= start_date_dt) & (df['Tanggal'] <= end_date_dt)].copy()
if df_filtered.empty: st.error("Tidak ada data di rentang tanggal yang dipilih."); st.stop()

my_store_name = "DB KLIK"
df_filtered['Minggu'] = df_filtered['Tanggal'].dt.to_period('W-SUN').apply(lambda p: p.start_time).dt.date
latest_entries_weekly = df_filtered.loc[df_filtered.groupby(['Minggu', 'Toko', 'Nama Produk'])['Tanggal'].idxmax()]
latest_entries_overall = df_filtered.loc[df_filtered.groupby(['Toko', 'Nama Produk'])['Tanggal'].idxmax()]
main_store_latest_overall = latest_entries_overall[latest_entries_overall['Toko'] == my_store_name]
competitor_latest_overall = latest_entries_overall[latest_entries_overall['Toko'] != my_store_name]
main_store_df = df_filtered[df_filtered['Toko'] == my_store_name]

# =========================================================================================
# ================================ TAMPILAN KONTEN UTAMA ================================
# =========================================================================================

if app_mode == "Tab Analisis":
    st.header("📈 Analisis Penjualan & Kompetitor")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⭐ Toko Saya", "⚖️ Perbandingan Cerdas", "🏆 Brand Kompetitor", "📦 Status Stok", "📈 Kinerja Penjualan", "📊 Produk Baru"])
    
    with tab1:
        st.subheader(f"Analisis Kinerja Toko: {my_store_name}")
        
        st.markdown("#### Produk Terlaris (Berdasarkan Unit Terjual)")
        top_products = main_store_latest_overall.sort_values('Terjual per Bulan', ascending=False).head(15).copy()
        for col in ['Harga', 'Omzet']:
            if col in top_products.columns: top_products[col] = top_products[col].apply(format_rupiah)
        st.dataframe(top_products[['Nama Produk', 'SKU', 'Harga', 'Terjual per Bulan', 'Omzet']], use_container_width=True, hide_index=True)
        
        st.markdown("#### Distribusi Omzet per Brand")
        brand_omzet_main = main_store_latest_overall.groupby('Brand')['Omzet'].sum().reset_index()
        if not brand_omzet_main.empty:
            fig_brand_pie = px.pie(brand_omzet_main.sort_values('Omzet', ascending=False).head(10), 
                                 names='Brand', values='Omzet', title='Distribusi Omzet Top 10 Brand')
            st.plotly_chart(fig_brand_pie, use_container_width=True)
        else:
            st.info("Tidak ada data omzet brand untuk ditampilkan.")

    with tab2:
        st.subheader(f"Perbandingan Produk '{my_store_name}' dengan Kompetitor")
        st.info("Pilih produk Anda untuk melihat perbandingannya. Data diambil dari hasil pembaruan terakhir.")
        if matches_df.empty:
            st.warning("Data perbandingan belum tersedia. Silakan jalankan pembaruan di sidebar."); st.stop()

        # --- PERBAIKAN KUNCI: Menambahkan 'penjaga' untuk format data lama ---
        required_match_cols = ['ID Produk Master (SKU)', 'Harga Master', 'Harga Kompetitor', 'Skor Kemiripan']
        if not all(col in matches_df.columns for col in required_match_cols):
            st.error("Format 'HASIL_MATCHING' sudah usang. Mohon jalankan 'Perbarui Perbandingan Sekarang' di sidebar untuk menggunakan mesin analisis baru.")
            st.stop()
        
        for col in ['Harga Master', 'Harga Kompetitor', 'Skor Kemiripan']:
            matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce')
        
        my_products_in_matches = main_store_latest_overall[main_store_latest_overall['SKU'].isin(matches_df['ID Produk Master (SKU)'].unique())]
        
        brand_list = sorted(my_products_in_matches['Brand'].unique())
        selected_brand = st.selectbox("Filter Brand:", ["Semua Brand"] + brand_list)
        
        products_to_show = my_products_in_matches[my_products_in_matches['Brand'] == selected_brand] if selected_brand != "Semua Brand" else my_products_in_matches
        product_list = sorted(products_to_show['Nama Produk'].unique())
        
        if not product_list:
             st.info("Tidak ada produk yang cocok dengan filter. Coba jalankan pembaruan."); st.stop()
             
        selected_product_name = st.selectbox("Pilih Produk Anda:", product_list, label_visibility="collapsed")

        if selected_product_name:
            try:
                selected_sku = products_to_show.loc[products_to_show['Nama Produk'] == selected_product_name, 'SKU'].iloc[0]
            except IndexError: st.error("Produk tidak ditemukan. Mungkin data belum sinkron."); st.stop()

            matches_for_product = matches_df[(matches_df['ID Produk Master (SKU)'] == selected_sku) & (matches_df['Skor Kemiripan'] >= accuracy_cutoff)].copy()
            my_product_info_row = main_store_latest_overall[main_store_latest_overall['SKU'] == selected_sku]
            
            if my_product_info_row.empty:
                st.error("Info produk Anda tidak ditemukan di data terbaru."); st.stop()

            my_price = my_product_info_row.iloc[0]['Harga']
            st.markdown(f"**Produk Pilihan:** *{my_product_info_row.iloc[0]['Nama Produk']}*")

            if matches_for_product.empty:
                st.warning("Tidak ada data perbandingan untuk produk ini dengan tingkat akurasi yang dipilih.")
            else:
                col1, col2 = st.columns(2)
                harga_terendah = matches_for_product['Harga Kompetitor'].min()
                delta = my_price - harga_terendah
                col1.metric("Harga Kompetitor Terendah", format_rupiah(harga_terendah), delta=f"Rp {delta:,.0f}" if delta !=0 else "Sama", delta_color="inverse")
                ready_count = matches_for_product[matches_for_product['Status Stok Kompetitor'] == 'Tersedia']['Toko Kompetitor'].nunique()
                col2.metric("Jumlah Kompetitor (Stok Ready)", f"{ready_count} Toko")
                st.divider()

                display_list = [{'Toko': f"{my_store_name} (Anda)", 'Nama Produk Tercantum': my_product_info_row.iloc[0]['Nama Produk'], 'Harga': my_price,
                                 'Selisih Harga': "Rp 0 (Basis)", 'Status Stok': "Tersedia", 'Skor Kemiripan (%)': 100}]
                for _, match in matches_for_product.iterrows():
                    comp_price = match['Harga Kompetitor']
                    price_diff = comp_price - my_price
                    diff_text = " (Sama)" if price_diff == 0 else " (Lebih Mahal)" if price_diff > 0 else " (Lebih Murah)"
                    display_list.append({'Toko': match['Toko Kompetitor'], 'Nama Produk Tercantum': match['Produk Kompetitor'], 'Harga': comp_price,
                                         'Selisih Harga': f"Rp {price_diff:,.0f}{diff_text}", 'Status Stok': match['Status Stok Kompetitor'],
                                         'Skor Kemiripan (%)': match['Skor Kemiripan']})
                
                display_df = pd.DataFrame(display_list).sort_values(by='Harga').reset_index(drop=True)
                display_df['Harga'] = display_df['Harga'].apply(format_rupiah)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
    with tab3:
        st.subheader("Analisis Brand di Toko Kompetitor")
        st.info("Snapshot omzet brand teratas di setiap toko kompetitor.")
        for store in sorted(competitor_latest_overall['Toko'].unique()):
            with st.expander(f"Analisis untuk: **{store}**"):
                store_df = competitor_latest_overall[competitor_latest_overall['Toko'] == store]
                brand_omzet = store_df.groupby('Brand')['Omzet'].sum().nlargest(10).reset_index()
                if not brand_omzet.empty:
                    fig = px.bar(brand_omzet, x='Brand', y='Omzet', title=f"Top 10 Brand di {store}", text_auto='.2s')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Tidak ada data omzet untuk toko ini.")

    with tab4:
        st.subheader("Tren Status Stok Mingguan per Toko")
        stock_trends = df_filtered.groupby(['Minggu', 'Toko', 'Status']).size().unstack(fill_value=0).reset_index()
        stock_trends_melted = stock_trends.melt(id_vars=['Minggu', 'Toko'], value_vars=['Tersedia', 'Habis'], var_name='Tipe Stok', value_name='Jumlah Produk')
        fig = px.line(stock_trends_melted, x='Minggu', y='Jumlah Produk', color='Toko', line_dash='Tipe Stok', markers=True, title='Jumlah Produk Tersedia vs. Habis per Minggu')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab5:
        st.subheader("Kinerja Penjualan Mingguan Antar Toko")
        weekly_omzet = df_filtered.groupby(['Minggu', 'Toko'])['Omzet'].sum().reset_index()
        fig = px.line(weekly_omzet, x='Minggu', y='Omzet', color='Toko', markers=True, title='Perbandingan Omzet Mingguan')
        fig.update_layout(yaxis_title="Total Omzet")
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("Analisis Produk Baru Mingguan")
        weeks = sorted(df_filtered['Minggu'].unique())
        if len(weeks) < 2:
            st.info("Butuh setidaknya 2 minggu data untuk melakukan perbandingan.")
        else:
            col1, col2 = st.columns(2)
            week_before = col1.selectbox("Pilih Minggu Pembanding:", weeks, index=len(weeks)-2 if len(weeks) > 1 else 0)
            week_after = col2.selectbox("Pilih Minggu Penentu:", weeks, index=len(weeks)-1)
            if week_before >= week_after:
                st.error("Minggu Penentu harus setelah Minggu Pembanding.")
            else:
                for store in sorted(df_filtered['Toko'].unique()):
                    with st.expander(f"Produk Baru di Toko: **{store}**"):
                        products_before = set(df_filtered[(df_filtered['Toko'] == store) & (df_filtered['Minggu'] == week_before)]['Nama Produk'])
                        products_after = set(df_filtered[(df_filtered['Toko'] == store) & (df_filtered['Minggu'] == week_after)]['Nama Produk'])
                        new_products = products_after - products_before
                        if not new_products:
                            st.write("Tidak ada produk baru yang terdeteksi.")
                        else:
                            st.write(f"Ditemukan **{len(new_products)}** produk baru:")
                            new_products_df = latest_entries_overall[(latest_entries_overall['Nama Produk'].isin(new_products)) & (latest_entries_overall['Toko'] == store)].copy()
                            new_products_df['Harga'] = new_products_df['Harga'].apply(format_rupiah)
                            st.dataframe(new_products_df[['Nama Produk', 'Harga', 'Brand']], use_container_width=True, hide_index=True)

elif app_mode == "HPP Produk":
    st.header("💰 Analisis Harga Pokok Penjualan (HPP)")
    if db_df.empty or 'SKU' not in db_df.columns:
        st.error("Sheet 'DATABASE' tidak ditemukan atau tidak valid untuk analisis HPP."); st.stop()

    db_df['HPP'] = pd.to_numeric(db_df.get('HPP (LATEST)'), errors='coerce').fillna(pd.to_numeric(db_df.get('HPP (AVERAGE)'), errors='coerce'))
    hpp_data = db_df[['SKU', 'HPP']].dropna(subset=['SKU', 'HPP'])
    hpp_data = hpp_data[hpp_data['SKU'].astype(str) != ''].drop_duplicates(subset=['SKU'])
    
    merged_df = pd.merge(main_store_latest_overall, hpp_data, on='SKU', how='left')
    merged_df['Selisih'] = merged_df['Harga'] - merged_df['HPP']

    df_rugi = merged_df[merged_df['Selisih'] < 0].copy()
    df_untung = merged_df[merged_df['Selisih'] >= 0].copy()
    df_tidak_ditemukan = merged_df[merged_df['HPP'].isnull()].copy()
    
    st.subheader("🔴 Produk Dijual di Bawah HPP")
    if df_rugi.empty:
        st.success("👍 Tidak ada produk yang dijual di bawah HPP.")
    else:
        df_rugi_display = df_rugi[['Nama Produk', 'SKU', 'Harga', 'HPP', 'Selisih']].copy()
        for col in ['Harga', 'HPP', 'Selisih']: df_rugi_display[col] = df_rugi_display[col].apply(format_rupiah)
        st.dataframe(df_rugi_display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🟢 Produk Dijual di Atas HPP")
    if df_untung.empty:
        st.warning("Tidak ada produk yang dijual di atas HPP.")
    else:
        df_untung_display = df_untung[['Nama Produk', 'SKU', 'Harga', 'HPP', 'Selisih']].copy()
        for col in ['Harga', 'HPP', 'Selisih']: df_untung_display[col] = df_untung_display[col].apply(format_rupiah)
        st.dataframe(df_untung_display, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("❓ Produk Tanpa Data HPP")
    if df_tidak_ditemukan.empty:
        st.success("Semua produk berhasil dicocokkan dengan data HPP.")
    else:
        st.warning("Produk berikut tidak memiliki data HPP di sheet 'DATABASE'.")
        df_na_display = df_tidak_ditemukan[['Nama Produk', 'SKU', 'Harga']].copy()
        df_na_display['Harga'] = df_na_display['Harga'].apply(format_rupiah)
        st.dataframe(df_na_display, use_container_width=True, hide_index=True)

