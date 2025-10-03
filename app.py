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
    st.header("📈 Tampilan Analisis Penjualan & Kompetitor")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⭐ Analisis Toko Saya", "⚖️ Perbandingan Harga", "🏆 Analisis Brand Kompetitor", "📦 Status Stok Produk", "📈 Kinerja Penjualan", "📊 Analisis Mingguan"])
    
    # KODE UNTUK SEMUA TAB DARI VERSI SEBELUMNYA TETAP SAMA DI SINI
    with tab1:
        st.header(f"Analisis Kinerja Toko: {my_store_name}")
        
        section_counter = 1

        st.subheader(f"{section_counter}. Analisis Kategori Terlaris (Berdasarkan Omzet)")
        section_counter += 1
        
        if 'KATEGORI' in main_store_latest_overall.columns:
            main_store_cat = main_store_latest_overall.copy()
            main_store_cat['KATEGORI'] = main_store_cat['KATEGORI'].replace('', 'Lainnya').fillna('Lainnya')
            
            category_sales = main_store_cat.groupby('KATEGORI')['Omzet'].sum().reset_index()
            
            if not category_sales.empty:
                cat_sales_sorted = category_sales.sort_values('Omzet', ascending=False).head(10)
                fig_cat = px.bar(cat_sales_sorted, x='KATEGORI', y='Omzet', title='Top 10 Kategori Berdasarkan Omzet', text_auto='.2s')
                st.plotly_chart(fig_cat, use_container_width=True)

                st.markdown("##### Rincian Data Omzet per Kategori")
                table_cat_sales = cat_sales_sorted.copy()
                table_cat_sales['Omzet'] = table_cat_sales['Omzet'].apply(lambda x: f"Rp {int(x):,.0f}")
                st.dataframe(table_cat_sales, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Lihat Produk Terlaris per Kategori")
                
                category_list = category_sales.sort_values('Omzet', ascending=False)['KATEGORI'].tolist()
                
                selected_category = st.selectbox(
                    "Pilih Kategori untuk melihat produk terlaris:",
                    options=category_list
                )

                if selected_category:
                    products_in_category = main_store_cat[main_store_cat['KATEGORI'] == selected_category].copy()
                    top_products_in_category = products_in_category.sort_values('Terjual per Bulan', ascending=False)

                    if top_products_in_category.empty:
                        st.info(f"Tidak ada produk terlaris untuk kategori '{selected_category}'.")
                    else:
                        columns_to_display = ['Nama Produk', 'SKU', 'Harga', 'Terjual per Bulan', 'Omzet']
                        if 'SKU' not in top_products_in_category.columns:
                            top_products_in_category['SKU'] = 'N/A'
                        
                        display_table = top_products_in_category[columns_to_display].copy()
                        display_table['Harga'] = display_table['Harga'].apply(lambda x: f"Rp {int(x):,.0f}")
                        display_table['Omzet'] = display_table['Omzet'].apply(lambda x: f"Rp {int(x):,.0f}")
                        
                        st.dataframe(display_table, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada data omzet per kategori untuk ditampilkan.")
        else:
            st.warning("Kolom 'KATEGORI' tidak ditemukan pada data toko Anda. Analisis ini dilewati.")

        st.subheader(f"{section_counter}. Produk Terlaris")
        section_counter += 1
        top_products = main_store_latest_overall.sort_values('Terjual per Bulan', ascending=False).head(15).copy()
        top_products['Harga_rp'] = top_products['Harga'].apply(lambda x: f"Rp {int(x):,.0f}")
        top_products['Omzet_rp'] = top_products['Omzet'].apply(lambda x: f"Rp {int(x):,.0f}")
        
        display_cols_top = ['Nama Produk', 'SKU', 'Harga_rp', 'Omzet_rp', 'Terjual per Bulan']
        if 'SKU' not in top_products.columns:
            top_products['SKU'] = 'N/A'
        
        display_df_top = top_products[display_cols_top].rename(
            columns={'Harga_rp': 'Harga', 'Omzet_rp': 'Omzet'}
        )
        st.dataframe(display_df_top, use_container_width=True, hide_index=True)

        st.subheader(f"{section_counter}. Distribusi Omzet Brand")
        section_counter += 1
        brand_omzet_main = main_store_latest_overall.groupby('Brand')['Omzet'].sum().reset_index()
        if not brand_omzet_main.empty:
            fig_brand_pie = px.pie(brand_omzet_main.sort_values('Omzet', ascending=False).head(7), 
                                names='Brand', values='Omzet', title='Distribusi Omzet Top 7 Brand (Snapshot Terakhir)')
            
            fig_brand_pie.update_traces(
                textposition='outside',
                texttemplate='%{label}<br><b>Rp %{value:,.0f}</b><br>(%{percent})',
                insidetextfont=dict(color='white')
            )
            fig_brand_pie.update_layout(showlegend=False)
            
            st.plotly_chart(fig_brand_pie, use_container_width=True)
        else:
            st.info("Tidak ada data omzet brand.")

        st.subheader(f"{section_counter}. Ringkasan Kinerja Mingguan (WoW Growth)")
        section_counter += 1
        main_store_latest_weekly = main_store_df.loc[main_store_df.groupby(['Minggu', 'Nama Produk'])['Tanggal'].idxmax()]
        weekly_summary_tab1 = main_store_latest_weekly.groupby('Minggu').agg(
            Omzet=('Omzet', 'sum'), Penjualan_Unit=('Terjual per Bulan', 'sum')
        ).reset_index().sort_values('Minggu')
        weekly_summary_tab1['Pertumbuhan Omzet (WoW)'] = weekly_summary_tab1['Omzet'].pct_change().apply(format_wow_growth)
        weekly_summary_tab1['Omzet'] = weekly_summary_tab1['Omzet'].apply(lambda x: f"Rp {x:,.0f}")
        st.dataframe(
            weekly_summary_tab1[['Minggu', 'Omzet', 'Penjualan_Unit', 'Pertumbuhan Omzet (WoW)']].style.applymap(
                style_wow_growth, subset=['Pertumbuhan Omzet (WoW)']
            ), use_container_width=True, hide_index=True
        )

    with tab2:
        st.subheader(f"Perbandingan Produk '{my_store_name}' dengan Kompetitor")
        st.info("Pilih produk Anda untuk melihat perbandingannya. Data diambil dari hasil pembaruan terakhir.")
        if matches_df.empty:
            st.warning("Data perbandingan belum tersedia. Silakan jalankan pembaruan di sidebar."); st.stop()

        for col in ['Harga Master', 'Harga Kompetitor', 'Skor Kemiripan']:
            if col in matches_df.columns: matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce')
        
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
        st.header("Analisis Brand di Toko Kompetitor")
        if competitor_df.empty:
            st.warning("Tidak ada data kompetitor pada rentang tanggal ini.")
        else:
            competitor_list = sorted(competitor_df['Toko'].unique())
            for competitor_store in competitor_list:
                with st.expander(f"Analisis untuk Kompetitor: **{competitor_store}**"):
                    single_competitor_df = competitor_latest_overall[competitor_latest_overall['Toko'] == competitor_store]
                    brand_analysis = single_competitor_df.groupby('Brand').agg(
                        Total_Omzet=('Omzet', 'sum'), 
                        Total_Unit_Terjual=('Terjual per Bulan', 'sum')
                    ).reset_index().sort_values("Total_Omzet", ascending=False)
                    
                    if not brand_analysis.empty:
                        display_brand_analysis = brand_analysis.head(10).copy()
                        display_brand_analysis['Total_Omzet'] = display_brand_analysis['Total_Omzet'].apply(lambda x: f"Rp {int(x):,.0f}")
                        st.dataframe(display_brand_analysis, use_container_width=True, hide_index=True)

                        fig_pie_comp = px.pie(brand_analysis.head(7), names='Brand', values='Total_Omzet', title=f'Distribusi Omzet Top 7 Brand di {competitor_store} (Snapshot Terakhir)')
                        st.plotly_chart(fig_pie_comp, use_container_width=True)
                    else:
                        st.info("Tidak ada data brand untuk toko ini.")

    with tab4:
        st.header("Tren Status Stok Mingguan per Toko")
        stock_trends = df_filtered.groupby(['Minggu', 'Toko', 'Status']).size().unstack(fill_value=0).reset_index()
        if 'Tersedia' not in stock_trends.columns: stock_trends['Tersedia'] = 0
        if 'Habis' not in stock_trends.columns: stock_trends['Habis'] = 0
        stock_trends_melted = stock_trends.melt(id_vars=['Minggu', 'Toko'], value_vars=['Tersedia', 'Habis'], var_name='Tipe Stok', value_name='Jumlah Produk')
        
        fig_stock_trends = px.line(stock_trends_melted, x='Minggu', y='Jumlah Produk', color='Toko', line_dash='Tipe Stok', markers=True, title='Jumlah Produk Tersedia vs. Habis per Minggu')
        st.plotly_chart(fig_stock_trends, use_container_width=True)
        st.dataframe(stock_trends.set_index('Minggu'), use_container_width=True)

    with tab5:
        st.header("Analisis Kinerja Penjualan (Semua Toko)")
        
        all_stores_latest_per_week = latest_entries_weekly.groupby(['Minggu', 'Toko'])['Omzet'].sum().reset_index()
        fig_weekly_omzet = px.line(all_stores_latest_per_week, x='Minggu', y='Omzet', color='Toko', markers=True, title='Perbandingan Omzet Mingguan Antar Toko (Berdasarkan Snapshot Terakhir)')
        st.plotly_chart(fig_weekly_omzet, use_container_width=True)
        
        st.subheader("Tabel Rincian Omzet per Tanggal")
        if not df_filtered.empty:
            omzet_pivot = df_filtered.pivot_table(index='Toko', columns='Tanggal', values='Omzet', aggfunc='sum').fillna(0)
            omzet_pivot.columns = [col.strftime('%d %b %Y') for col in omzet_pivot.columns]
            for col in omzet_pivot.columns:
                omzet_pivot[col] = omzet_pivot[col].apply(lambda x: f"Rp {int(x):,}" if x > 0 else "-")
            omzet_pivot.reset_index(inplace=True)
            st.info("Anda bisa scroll tabel ini ke samping untuk melihat tanggal lainnya.")
            st.dataframe(omzet_pivot, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada data untuk ditampilkan dalam tabel.")

    with tab6:
        st.header("Analisis Produk Baru Mingguan")
        weeks = sorted(df_filtered['Minggu'].unique())
        if len(weeks) < 2:
            st.info("Butuh setidaknya 2 minggu data untuk melakukan perbandingan produk baru.")
        else:
            col1, col2 = st.columns(2)
            week_before = col1.selectbox("Pilih Minggu Pembanding:", weeks, index=0)
            week_after = col2.selectbox("Pilih Minggu Penentu:", weeks, index=len(weeks)-1)

            if week_before >= week_after:
                st.error("Minggu Penentu harus setelah Minggu Pembanding.")
            else:
                all_stores = sorted(df_filtered['Toko'].unique())
                for store in all_stores:
                    with st.expander(f"Lihat Produk Baru di Toko: **{store}**"):
                        products_before = set(df_filtered[(df_filtered['Toko'] == store) & (df_filtered['Minggu'] == week_before) & (df_filtered['Status'] == 'Tersedia')]['Nama Produk'])
                        products_after = set(df_filtered[(df_filtered['Toko'] == store) & (df_filtered['Minggu'] == week_after) & (df_filtered['Status'] == 'Tersedia')]['Nama Produk'])
                        new_products = products_after - products_before
                        
                        if not new_products:
                            st.write("Tidak ada produk baru yang terdeteksi.")
                        else:
                            st.write(f"Ditemukan **{len(new_products)}** produk baru:")
                            new_products_df = df_filtered[df_filtered['Nama Produk'].isin(new_products) & (df_filtered['Toko'] == store) & (df_filtered['Minggu'] == week_after)].copy()
                            new_products_df['Harga_fmt'] = new_products_df['Harga'].apply(lambda x: f"Rp {int(x):,.0f}")
                            st.dataframe(new_products_df[['Nama Produk', 'Harga_fmt', 'Stok', 'Brand']].rename(columns={'Harga_fmt':'Harga'}), use_container_width=True, hide_index=True)

elif app_mode == "HPP Produk":
    st.header("💰 Tampilan Analisis Harga Pokok Penjualan (HPP)")

    # 1. PERSIAPAN DATA HPP DARI SHEET 'DATABASE'
    if db_df.empty or 'SKU' not in db_df.columns:
        st.error("Sheet 'DATABASE' tidak ditemukan atau tidak memiliki kolom 'SKU'. Analisis HPP tidak dapat dilanjutkan.")
        st.stop()

    # Pastikan kolom HPP ada dan bersihkan
    if 'HPP (LATEST)' not in db_df.columns: db_df['HPP (LATEST)'] = np.nan
    if 'HPP (AVERAGE)' not in db_df.columns: db_df['HPP (AVERAGE)'] = np.nan

    # Konversi kolom HPP ke numerik, paksa error menjadi NaN (kosong)
    db_df['HPP_LATEST_NUM'] = pd.to_numeric(db_df['HPP (LATEST)'], errors='coerce')
    db_df['HPP_AVERAGE_NUM'] = pd.to_numeric(db_df['HPP (AVERAGE)'], errors='coerce')

    # Logika fallback: Gunakan LATEST, jika kosong, gunakan AVERAGE
    db_df['HPP'] = db_df['HPP_LATEST_NUM'].fillna(db_df['HPP_AVERAGE_NUM'])
    
    # Pilih hanya data HPP yang relevan dan bersih
    hpp_data = db_df[['SKU', 'HPP']].copy()
    hpp_data.dropna(subset=['SKU', 'HPP'], inplace=True)
    hpp_data = hpp_data[hpp_data['SKU'] != '']
    hpp_data.drop_duplicates(subset=['SKU'], keep='first', inplace=True)

    # 2. GABUNGKAN DATA PENJUALAN TERBARU DENGAN DATA HPP
    # Menggunakan data penjualan terbaru dari toko Anda
    latest_db_klik = main_store_latest_overall.copy()
    
    # Gabungkan berdasarkan SKU. `how='left'` menjaga semua produk dari toko Anda.
    merged_df = pd.merge(latest_db_klik, hpp_data, on='SKU', how='left')

    # 3. HITUNG SELISIH DAN PISAHKAN DATA
    merged_df['Selisih'] = merged_df['Harga'] - merged_df['HPP']

    # Tabel 1: Jual lebih murah dari HPP (Rugi)
    df_rugi = merged_df[merged_df['Selisih'] < 0].copy()

    # Tabel 2: Jual lebih mahal dari HPP (Untung)
    df_untung = merged_df[(merged_df['Selisih'] >= 0)].copy()

    # Tabel 3: Produk tidak ditemukan HPP-nya di DATABASE
    df_tidak_ditemukan = merged_df[merged_df['HPP'].isnull()].copy()

    # 4. TAMPILKAN TABEL-TABEL HASIL ANALISIS
    
    # --- TABEL 1: PRODUK DIJUAL DI BAWAH HPP ---
    st.subheader("🔴 Produk Lebih Murah dari HPP")
    if df_rugi.empty:
        st.success("👍 Mantap! Tidak ada produk yang dijual di bawah HPP.")
    else:
        display_rugi = df_rugi[['Nama Produk', 'SKU', 'Harga', 'HPP', 'Selisih', 'Terjual per Bulan', 'Omzet']].copy()
        display_rugi.rename(columns={'Terjual per Bulan': 'Terjual/Bln'}, inplace=True)
        # Formatting
        for col in ['Harga', 'HPP', 'Selisih', 'Omzet']:
            display_rugi[col] = display_rugi[col].apply(format_rupiah)
        st.dataframe(display_rugi, use_container_width=True, hide_index=True)

    st.divider()

    # --- TABEL 2: PRODUK DIJUAL DI ATAS HPP ---
    st.subheader("🟢 Produk Lebih Mahal dari HPP")
    if df_untung.empty:
        st.warning("Tidak ada produk yang dijual di atas HPP.")
    else:
        display_untung = df_untung[['Nama Produk', 'SKU', 'Harga', 'HPP', 'Selisih', 'Terjual per Bulan', 'Omzet']].copy()
        display_untung.rename(columns={'Terjual per Bulan': 'Terjual/Bln'}, inplace=True)
        # Formatting
        for col in ['Harga', 'HPP', 'Selisih', 'Omzet']:
            display_untung[col] = display_untung[col].apply(format_rupiah)
        st.dataframe(display_untung, use_container_width=True, hide_index=True)

    st.divider()
    
    # --- TABEL 3: PRODUK TIDAK TERDETEKSI ---
    st.subheader("❓ Produk Tidak Terdeteksi HPP-nya")
    if df_tidak_ditemukan.empty:
        st.success("👍 Semua produk yang dijual berhasil dicocokkan dengan data HPP di DATABASE.")
    else:
        st.warning("Mohon untuk mengecek data produk lagi, sepertinya ada data yang tidak akurat atau SKU tidak cocok.")
        display_tidak_ditemukan = df_tidak_ditemukan[['Nama Produk', 'SKU', 'Harga', 'Terjual per Bulan', 'Omzet']].copy()
        display_tidak_ditemukan.rename(columns={'Terjual per Bulan': 'Terjual/Bln'}, inplace=True)
        # Formatting
        for col in ['Harga', 'Omzet']:
            display_tidak_ditemukan[col] = display_tidak_ditemukan[col].apply(format_rupiah)
        st.dataframe(display_tidak_ditemukan, use_container_width=True, hide_index=True)
