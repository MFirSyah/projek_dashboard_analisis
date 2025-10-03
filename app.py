# ===================================================================================
#  DASHBOARD ANALISIS & TOOLS - VERSI 7.3 (FINAL INTEGRATION)
#  Dibuat oleh: Firman & Asisten AI Gemini
#  Versi ini memperbaiki NameError dan melengkapi semua fungsionalitas tab
#  dalam struktur menu yang terintegrasi.
# ===================================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import re
import gspread
from datetime import datetime
from gspread_dataframe import set_with_dataframe
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis & Tools")

# ================================
# FUNGSI-FUNGSI INTI (Koneksi, Normalisasi, dll.)
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
    stopwords = ['garansi', 'resmi', 'original', 'dan', 'promo', 'murah', 'untuk', 'dengan', 'built', 'in', 'speaker', 'hdmi', 'dp', 'vga', 'ms', 'office']
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

@st.cache_data(ttl=600, show_spinner="Memuat data dari Google Sheets...")
def load_all_data(spreadsheet_key):
    gc = connect_to_gsheets()
    try:
        spreadsheet = gc.open_by_key(spreadsheet_key)
    except Exception as e:
        st.error(f"GAGAL KONEKSI/OPEN SPREADSHEET: {e}"); return None, None
    
    all_sheets = spreadsheet.worksheets()
    rekap_list_df, database_df = [], pd.DataFrame()

    for worksheet in all_sheets:
        sheet_name = worksheet.title
        try:
            if "DATABASE" == sheet_name.upper():
                database_df = pd.DataFrame(worksheet.get_all_records(head=1))
            elif "REKAP" in sheet_name.upper():
                all_values = worksheet.get_all_values()
                if not all_values or len(all_values) < 2: continue
                df_sheet = pd.DataFrame(data=all_values[1:], columns=all_values[0])
                store_name_match = re.match(r"^(.*?) - REKAP", sheet_name, re.IGNORECASE)
                df_sheet['Toko'] = store_name_match.group(1).strip() if store_name_match else "Toko Tak Dikenal"
                df_sheet['Status'] = 'Tersedia' if "READY" in sheet_name.upper() else 'Habis'
                rekap_list_df.append(df_sheet)
        except Exception as e:
            st.warning(f"Gagal memproses sheet '{sheet_name}': {e}")

    if not rekap_list_df: return None, None
    rekap_df = pd.concat(rekap_list_df, ignore_index=True)
    
    rekap_df.columns = [str(c).strip().upper() for c in rekap_df.columns]
    final_rename = {'NAMA': 'Nama Produk', 'TERJUAL/BLN': 'Terjual per Bulan', 'TANGGAL': 'Tanggal', 'HARGA': 'Harga', 'BRAND': 'Brand', 'STATUS': 'Status', 'KATEGORI': 'Kategori', 'SKU': 'SKU', 'TOKO': 'Toko'}
    rekap_df.rename(columns=final_rename, inplace=True)
    
    required_cols = ['Tanggal', 'Nama Produk', 'Harga', 'Toko']
    missing_cols = [col for col in required_cols if col not in rekap_df.columns]
    if missing_cols:
        st.error(f"Kolom wajib berikut tidak ditemukan: {', '.join(missing_cols)}.")
        return None, None
        
    rekap_df['Tanggal'] = pd.to_datetime(rekap_df['Tanggal'], errors='coerce', dayfirst=True)
    rekap_df['Harga'] = pd.to_numeric(rekap_df['Harga'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')
    rekap_df['Terjual per Bulan'] = pd.to_numeric(rekap_df.get('Terjual per Bulan'), errors='coerce').fillna(0)
    rekap_df.dropna(subset=required_cols, inplace=True)
    
    if 'Brand' not in rekap_df.columns: rekap_df['Brand'] = rekap_df['Nama Produk'].str.split().str[0]
    rekap_df['Brand'].fillna("LAINNYA", inplace=True)
    rekap_df['Omzet'] = (rekap_df['Harga'].fillna(0) * rekap_df['Terjual per Bulan'].fillna(0)).astype(int)
    
    return rekap_df.sort_values('Tanggal'), database_df

def format_rupiah(val):
    if pd.isna(val) or not isinstance(val, (int, float, np.number)): return "N/A"
    return f"Rp {int(val):,}"

# ================================
# FUNGSI UNTUK ALAT SIMILARITY PRODUK
# ================================
def find_matches_for_similarity_tool(selected_product_row, my_store_df, competitor_df, score_cutoff=0.6):
    selected_brand = selected_product_row['Brand']
    
    competitor_filtered = competitor_df[competitor_df['Brand'] == selected_brand].copy()
    
    results = [{'Nama Produk Tercantum': selected_product_row['Nama Produk'], 'Toko': 'DB KLIK (Anda)', 'Harga': selected_product_row['Harga'], 'Status Stok': selected_product_row['Status'], 'Skor Kemiripan (%)': 100}]
    if competitor_filtered.empty: 
        return results

    all_names = pd.concat([my_store_df['Nama Normalisasi'], competitor_filtered['Nama Normalisasi']]).unique()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
    vectorizer.fit(all_names)

    vec_selected = vectorizer.transform([selected_product_row['Nama Normalisasi']])
    vec_competitor = vectorizer.transform(competitor_filtered['Nama Normalisasi'])
    similarities = cosine_similarity(vec_selected, vec_competitor)[0]

    for i, score in enumerate(similarities):
        if score >= score_cutoff:
            match = competitor_filtered.iloc[i]
            results.append({'Nama Produk Tercantum': match['Nama Produk'], 'Toko': match['Toko'], 'Harga': match['Harga'], 'Status Stok': match['Status'], 'Skor Kemiripan (%)': int(score * 100)})
    return results

# ================================
# FUNGSI UNTUK ALAT LABELING (OPTIMIZED)
# ================================
def run_sku_category_labeling_optimized(gc, spreadsheet_key):
    placeholder = st.empty()
    with placeholder.container():
        st.info("Memulai proses labeling otomatis (Optimized)...")
        prog = st.progress(0, text="Langkah 1/4: Memuat data...")

    try:
        spreadsheet = gc.open_by_key(spreadsheet_key)
        db_sheet = spreadsheet.worksheet("DATABASE")
        db_klik_ready_sheet = spreadsheet.worksheet("DB KLIK - REKAP - READY")
        db_klik_habis_sheet = spreadsheet.worksheet("DB KLIK - REKAP - HABIS")

        database_df = pd.DataFrame(db_sheet.get_all_records(head=1))
        db_klik_ready_df = pd.DataFrame(db_klik_ready_sheet.get_all_records(head=1))
        db_klik_habis_df = pd.DataFrame(db_klik_habis_sheet.get_all_records(head=1))
    except Exception as e:
        with placeholder.container(): st.error(f"Gagal membuka worksheet: {e}"); return

    prog.progress(25, text="Langkah 2/4: Normalisasi teks...")
    database_df['Nama Normalisasi'] = database_df['NAMA'].apply(normalize_text)
    db_klik_ready_df['Nama Normalisasi'] = db_klik_ready_df['NAMA'].apply(normalize_text)
    db_klik_habis_df['Nama Normalisasi'] = db_klik_habis_df['NAMA'].apply(normalize_text)
    
    updates_ready, updates_habis = [], []
    
    db_klik_ready_df['sheet'] = 'ready'
    db_klik_habis_df['sheet'] = 'habis'
    combined_db_klik_df = pd.concat([db_klik_ready_df, db_klik_habis_df])
    
    all_brands = combined_db_klik_df['BRAND'].unique()
    total_brands = len(all_brands)
    prog.progress(50, text=f"Langkah 3/4: Memproses 0/{total_brands} brand...")

    for i, brand in enumerate(all_brands):
        prog.progress(50 + int((i / total_brands) * 40), text=f"Langkah 3/4: Memproses brand '{brand}' ({i+1}/{total_brands})...")
        
        db_brand_df = database_df[database_df['BRAND'] == brand]
        klik_brand_df = combined_db_klik_df[combined_db_klik_df['BRAND'] == brand]
        if db_brand_df.empty or klik_brand_df.empty: continue

        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
        all_names = pd.concat([db_brand_df['Nama Normalisasi'], klik_brand_df['Nama Normalisasi']]).unique()
        vectorizer.fit(all_names)
        db_vectors = vectorizer.transform(db_brand_df['Nama Normalisasi'])
        klik_vectors = vectorizer.transform(klik_brand_df['Nama Normalisasi'])
        
        similarities = cosine_similarity(klik_vectors, db_vectors)
        best_matches_indices = similarities.argmax(axis=1)

        try:
            sku_col_ready = db_klik_ready_df.columns.to_list().index('SKU') + 1
            kategori_col_ready = db_klik_ready_df.columns.to_list().index('KATEGORI') + 1
            sku_col_habis = db_klik_habis_df.columns.to_list().index('SKU') + 1
            kategori_col_habis = db_klik_habis_df.columns.to_list().index('KATEGORI') + 1
        except ValueError:
            with placeholder.container(): st.error("Kolom 'SKU' atau 'KATEGORI' tidak ditemukan."); return

        for j, match_idx in enumerate(best_matches_indices):
            original_row = klik_brand_df.iloc[j]
            best_match_db = db_brand_df.iloc[match_idx]
            
            row_index_in_sheet = original_row.name + 2
            
            if original_row['sheet'] == 'ready':
                updates_ready.append({'range': f'R{row_index_in_sheet}C{sku_col_ready}', 'values': [[best_match_db['SKU']]]})
                updates_ready.append({'range': f'R{row_index_in_sheet}C{kategori_col_ready}', 'values': [[best_match_db['KATEGORI']]]})
            else:
                updates_habis.append({'range': f'R{row_index_in_sheet}C{sku_col_habis}', 'values': [[best_match_db['SKU']]]})
                updates_habis.append({'range': f'R{row_index_in_sheet}C{kategori_col_habis}', 'values': [[best_match_db['KATEGORI']]]})
    
    prog.progress(95, text="Langkah 4/4: Menulis pembaruan ke Google Sheets...")
    try:
        with st.spinner("Menulis pembaruan... Ini mungkin memakan waktu beberapa saat."):
            if updates_ready:
                db_klik_ready_sheet.batch_update(updates_ready, value_input_option='USER_ENTERED')
            if updates_habis:
                db_klik_habis_sheet.batch_update(updates_habis, value_input_option='USER_ENTERED')
        prog.progress(100)
        with placeholder.container(): st.success(f"Labeling Selesai! {len(updates_ready)//2} produk READY dan {len(updates_habis)//2} produk HABIS telah diperbarui.")
        load_all_data.clear()
    except Exception as e:
        with placeholder.container(): st.error(f"Gagal menulis ke Google Sheets: {e}")

# ================================
# APLIKASI UTAMA (MAIN APP)
# ================================
st.title("📊 Dashboard Analisis & Tools")
try:
    SPREADSHEET_KEY = st.secrets["SOURCE_SPREADSHEET_ID"]
except KeyError:
    st.error("ID Spreadsheet belum diatur di Secrets. Mohon atur `SOURCE_SPREADSHEET_ID`."); st.stop()

gc = connect_to_gsheets()

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    if st.button("Tarik Data & Mulai Analisis 🚀", type="primary"):
        df_data, db_df_data = load_all_data(SPREADSHEET_KEY)
        if df_data is not None:
            st.session_state.df = df_data
            st.session_state.db_df = db_df_data
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Gagal memuat data utama. Silakan periksa pesan error di atas.")
    st.info("👆 Klik tombol untuk menarik semua data yang diperlukan untuk analisis.")
    st.stop()

df = st.session_state.df

st.sidebar.header("Menu Utama")
main_menu = st.sidebar.radio("Pilih Alat:", ("Dashboard Analisis", "Similarity Produk", "Tools (Peralatan)"))

if main_menu == "Dashboard Analisis":
    st.header("📈 Dashboard Analisis Penjualan & Kompetitor")
    st.sidebar.header("Filter Dashboard")
    min_date_val, max_date_val = df['Tanggal'].min().date(), df['Tanggal'].max().date()
    start_date, end_date = st.sidebar.date_input("Rentang Tanggal:", [min_date_val, max_date_val], min_value=min_date_val, max_value=max_date_val)
    if len((start_date, end_date)) != 2: st.sidebar.warning("Pilih 2 tanggal."); st.stop()
    
    start_date_dt, end_date_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df_filtered = df[(df['Tanggal'] >= start_date_dt) & (df['Tanggal'] <= end_date_dt)].copy()
    if df_filtered.empty: st.error("Tidak ada data di rentang tanggal yang dipilih."); st.stop()

    # --- PERBAIKAN: Mendefinisikan variabel yang dibutuhkan oleh semua tab ---
    my_store_name = "DB KLIK"
    df_filtered['Minggu'] = df_filtered['Tanggal'].dt.to_period('W-SUN').apply(lambda p: p.start_time).dt.date
    latest_entries_overall = df_filtered.loc[df_filtered.groupby(['Toko', 'Nama Produk'])['Tanggal'].idxmax()]
    main_store_latest_overall = latest_entries_overall[latest_entries_overall['Toko'] == my_store_name]
    competitor_latest_overall = latest_entries_overall[latest_entries_overall['Toko'] != my_store_name]
    latest_entries_weekly = df_filtered.loc[df_filtered.groupby(['Minggu', 'Toko', 'Nama Produk'])['Tanggal'].idxmax()]

    tab1, tab3, tab4, tab5, tab6 = st.tabs(["⭐ Toko Saya", "🏆 Brand Kompetitor", "📦 Status Stok", "📈 Kinerja Penjualan", "📊 Produk Baru"])
    
    with tab1:
        st.subheader(f"Analisis Kinerja Toko: {my_store_name}")
        st.markdown("#### Produk Terlaris (Berdasarkan Unit Terjual)")
        top_products = main_store_latest_overall.sort_values('Terjual per Bulan', ascending=False).head(15)
        st.dataframe(top_products[['Nama Produk', 'SKU', 'Harga', 'Terjual per Bulan', 'Omzet']].style.format({"Harga": format_rupiah, "Omzet": format_rupiah}), use_container_width=True, hide_index=False)
        
        st.markdown("#### Distribusi Omzet per Brand")
        brand_omzet_main = main_store_latest_overall.groupby('Brand')['Omzet'].sum().reset_index()
        if not brand_omzet_main.empty:
            fig_brand_pie = px.pie(brand_omzet_main.sort_values('Omzet', ascending=False).head(10), 
                                 names='Brand', values='Omzet', title='Distribusi Omzet Top 10 Brand')
            st.plotly_chart(fig_brand_pie, use_container_width=True)

    with tab3:
        st.subheader("Analisis Brand di Toko Kompetitor")
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
        weekly_omzet = latest_entries_weekly.groupby(['Minggu', 'Toko'])['Omzet'].sum().reset_index()
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

elif main_menu == "Similarity Produk":
    st.header("⚖️ Alat Analisis Similarity Produk")
    st.info("Alat ini menggunakan TF-IDF dan Validasi Brand untuk menemukan produk serupa.")
    st.sidebar.header("Filter Similarity")
    accuracy_cutoff = st.sidebar.slider("Tingkat Akurasi Minimum (%)", 50, 100, 65, 1)

    df['Nama Normalisasi'] = df['Nama Produk'].apply(normalize_text)
    
    my_store_df = df[df['Toko'] == "DB KLIK"].copy()
    my_store_df.dropna(subset=['SKU', 'Brand'], inplace=True)
    my_store_df.sort_values('Tanggal', ascending=True, inplace=True)
    my_store_df.drop_duplicates(subset='SKU', keep='last', inplace=True)
    
    competitor_df = df[df['Toko'] != "DB KLIK"].copy()
    
    brand_list = ["Semua Brand"] + sorted(my_store_df['Brand'].unique().tolist())
    selected_brand_filter = st.selectbox("Filter berdasarkan Brand:", brand_list)

    if selected_brand_filter != "Semua Brand":
        product_list_df = my_store_df[my_store_df['Brand'] == selected_brand_filter]
    else:
        product_list_df = my_store_df
        
    product_list = sorted(product_list_df['Nama Produk'].unique().tolist())

    if not product_list: st.warning("Tidak ada produk 'DB KLIK' yang cocok dengan filter brand."); st.stop()
    
    st.caption("Anda bisa mengetik di dalam kotak di bawah untuk mencari produk.")
    selected_name = st.selectbox("Pilih Produk Anda (DB KLIK):", product_list, label_visibility="collapsed")

    if st.button("🔍 Analisis Produk", type="primary"):
        if selected_name:
            selected_product_row = my_store_df[my_store_df['Nama Produk'] == selected_name].iloc[0]
            with st.spinner("Menganalisis kemiripan produk..."):
                matches = find_matches_for_similarity_tool(selected_product_row, my_store_df, competitor_df, score_cutoff=accuracy_cutoff/100)
            
            st.subheader("Hasil Analisis")
            display_df = pd.DataFrame(matches).sort_values(by='Skor Kemiripan (%)', ascending=False)
            if not display_df.empty:
                my_price = display_df.iloc[0]['Harga']
                display_df['Selisih'] = display_df['Harga'] - my_price
                def format_selisih(val):
                    if val == 0: return "Rp 0 (Basis)"
                    status = " (Lebih Mahal)" if val > 0 else " (Lebih Murah)"
                    return f"Rp {val:,.0f}{status}"
                display_df['Harga'] = display_df['Harga'].apply(format_rupiah)
                display_df['Selisih Harga'] = display_df['Selisih'].apply(format_selisih)
                st.dataframe(display_df[['Nama Produk Tercantum', 'Toko', 'Harga', 'Selisih Harga', 'Status Stok', 'Skor Kemiripan (%)']], use_container_width=True, hide_index=True)
                if len(display_df) <= 1:
                    st.warning("Tidak ditemukan produk yang cocok di kompetitor.")
            else:
                st.warning("Tidak ditemukan produk yang cocok.")

elif main_menu == "Tools (Peralatan)":
    st.header("🛠️ Tools (Peralatan)")
    st.subheader("Labeling SKU dan Kategori Otomatis")
    st.warning("PERHATIAN: Proses ini akan **menimpa (rewrite)** data SKU dan Kategori pada sheet `DB KLIK - REKAP - READY` dan `DB KLIK - REKAP - HABIS` secara permanen. Gunakan dengan hati-hati.")
    if st.button("🚀 Mulai Proses Labeling", type="primary"):
        run_sku_category_labeling_optimized(gc, SPREADSHEET_KEY)

