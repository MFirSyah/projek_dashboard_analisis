# ===================================================================================
#  DASHBOARD ANALISIS & TOOLS - VERSI 7.0 (INTEGRATED)
#  Dibuat oleh: Firman & Asisten AI Gemini
#  Versi ini mengintegrasikan tiga modul utama:
#  1. Dashboard Analisis Umum
#  2. Alat Similarity Produk (Standalone)
#  3. Alat Auto-Labeling SKU & Kategori
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
                database_df = pd.DataFrame(worksheet.get_all_records())
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
    final_rename = {'NAMA': 'Nama Produk', 'TERJUAL/BLN': 'Terjual per Bulan', 'TANGGAL': 'Tanggal', 'HARGA': 'Harga', 'BRAND': 'Brand', 'STATUS': 'Status', 'KATEGORI': 'Kategori', 'SKU': 'SKU'}
    rekap_df.rename(columns=final_rename, inplace=True)
    
    rekap_df['Tanggal'] = pd.to_datetime(rekap_df['Tanggal'], errors='coerce', dayfirst=True)
    rekap_df['Harga'] = pd.to_numeric(rekap_df['Harga'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')
    rekap_df['Terjual per Bulan'] = pd.to_numeric(rekap_df['Terjual per Bulan'], errors='coerce').fillna(0)
    rekap_df.dropna(subset=['Tanggal', 'Nama Produk', 'Harga', 'Toko'], inplace=True)
    
    if 'Brand' not in rekap_df.columns: rekap_df['Brand'] = rekap_df['Nama Produk'].str.split().str[0]
    rekap_df['Brand'].fillna("LAINNYA", inplace=True)

    rekap_df['Omzet'] = (rekap_df['Harga'].fillna(0) * rekap_df['Terjual per Bulan'].fillna(0)).astype(int)
    
    return rekap_df.sort_values('Tanggal'), database_df

def format_rupiah(val):
    if pd.isna(val): return "N/A"
    return f"Rp {int(val):,}"

# ================================
# FUNGSI UNTUK ALAT SIMILARITY PRODUK
# ================================
def find_matches_for_similarity_tool(selected_product_sku, my_store_df, competitor_df, score_cutoff=0.6):
    selected_product = my_store_df[my_store_df['SKU'] == selected_product_sku].iloc[0]
    selected_brand = selected_product['Brand']
    
    competitor_filtered = competitor_df[competitor_df['Brand'] == selected_brand].copy()
    if competitor_filtered.empty: 
        return [{'Nama Produk Tercantum': selected_product['Nama Produk'], 'Toko': 'DB KLIK (Anda)', 'Harga': selected_product['Harga'], 'Status Stok': 'Tersedia', 'Skor Kemiripan (%)': 100}]

    all_names = pd.concat([my_store_df['Nama Normalisasi'], competitor_filtered['Nama Normalisasi']]).unique()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
    vectorizer.fit(all_names)

    vec_selected = vectorizer.transform([selected_product['Nama Normalisasi']])
    vec_competitor = vectorizer.transform(competitor_filtered['Nama Normalisasi'])
    similarities = cosine_similarity(vec_selected, vec_competitor)[0]

    results = []
    my_price = selected_product['Harga']
    results.append({'Nama Produk Tercantum': selected_product['Nama Produk'], 'Toko': 'DB KLIK (Anda)', 'Harga': my_price, 'Status Stok': 'Tersedia', 'Skor Kemiripan (%)': 100})

    for i, score in enumerate(similarities):
        if score >= score_cutoff:
            match = competitor_filtered.iloc[i]
            results.append({'Nama Produk Tercantum': match['Nama Produk'], 'Toko': match['Toko'], 'Harga': match['Harga'], 'Status Stok': match['Status'], 'Skor Kemiripan (%)': int(score * 100)})
    return results

# ================================
# FUNGSI UNTUK ALAT LABELING
# ================================
def run_sku_category_labeling(gc, spreadsheet_key):
    placeholder = st.empty()
    with placeholder.container():
        st.info("Memulai proses labeling otomatis...")
        prog = st.progress(0, text="Langkah 1/5: Memuat data...")

    try:
        spreadsheet = gc.open_by_key(spreadsheet_key)
        db_sheet = spreadsheet.worksheet("DATABASE")
        db_klik_ready_sheet = spreadsheet.worksheet("DB KLIK - REKAP - READY")
        db_klik_habis_sheet = spreadsheet.worksheet("DB KLIK - REKAP - HABIS")

        database_df = pd.DataFrame(db_sheet.get_all_records())
        db_klik_ready_df = pd.DataFrame(db_klik_ready_sheet.get_all_records())
        db_klik_habis_df = pd.DataFrame(db_klik_habis_sheet.get_all_records())
    except Exception as e:
        with placeholder.container(): st.error(f"Gagal membuka worksheet: {e}"); return

    for df, name in [(database_df, "DATABASE"), (db_klik_ready_df, "DB KLIK READY"), (db_klik_habis_df, "DB KLIK HABIS")]:
        if "NAMA" not in df.columns:
            with placeholder.container(): st.error(f"Kolom 'NAMA' tidak ditemukan di sheet '{name}'."); return
    if "SKU" not in database_df.columns or "KATEGORI" not in database_df.columns:
        with placeholder.container(): st.error("Kolom 'SKU' atau 'KATEGORI' tidak ditemukan di sheet 'DATABASE'."); return
    
    prog.progress(20, text="Langkah 2/5: Normalisasi teks...")
    database_df['Nama Normalisasi'] = database_df['NAMA'].apply(normalize_text)
    db_klik_ready_df['Nama Normalisasi'] = db_klik_ready_df['NAMA'].apply(normalize_text)
    db_klik_habis_df['Nama Normalisasi'] = db_klik_habis_df['NAMA'].apply(normalize_text)

    prog.progress(40, text="Langkah 3/5: Membangun model TF-IDF...")
    all_names = pd.concat([database_df['Nama Normalisasi'], db_klik_ready_df['Nama Normalisasi'], db_klik_habis_df['Nama Normalisasi']]).unique()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
    vectorizer.fit(all_names)
    db_vectors = vectorizer.transform(database_df['Nama Normalisasi'])

    updates_ready, updates_habis = [], []

    prog.progress(60, text=f"Langkah 4/5: Mencocokkan {len(db_klik_ready_df)} produk 'READY'...")
    if not db_klik_ready_df.empty:
        ready_vectors = vectorizer.transform(db_klik_ready_df['Nama Normalisasi'])
        similarities_ready = cosine_similarity(ready_vectors, db_vectors)
        best_matches_indices_ready = similarities_ready.argmax(axis=1)
        
        try:
            # Menggunakan .get_loc() untuk mencari posisi kolom secara dinamis
            sku_col_ready = db_klik_ready_df.columns.get_loc('SKU') + 1
            kategori_col_ready = db_klik_ready_df.columns.get_loc('KATEGORI') + 1
        except KeyError:
            with placeholder.container(): st.error("Kolom 'SKU' atau 'KATEGORI' tidak ditemukan di sheet DB KLIK - REKAP - READY."); return

        for i, match_idx in enumerate(best_matches_indices_ready):
            best_match = database_df.iloc[match_idx]
            updates_ready.append({'range': f'R{i+2}C{sku_col_ready}', 'values': [[best_match['SKU']]]})
            updates_ready.append({'range': f'R{i+2}C{kategori_col_ready}', 'values': [[best_match['KATEGORI']]]})

    prog.progress(80, text=f"Langkah 5/5: Mencocokkan {len(db_klik_habis_df)} produk 'HABIS'...")
    if not db_klik_habis_df.empty:
        habis_vectors = vectorizer.transform(db_klik_habis_df['Nama Normalisasi'])
        similarities_habis = cosine_similarity(habis_vectors, db_vectors)
        best_matches_indices_habis = similarities_habis.argmax(axis=1)

        try:
            sku_col_habis = db_klik_habis_df.columns.get_loc('SKU') + 1
            kategori_col_habis = db_klik_habis_df.columns.get_loc('KATEGORI') + 1
        except KeyError:
            with placeholder.container(): st.error("Kolom 'SKU' atau 'KATEGORI' tidak ditemukan di sheet DB KLIK - REKAP - HABIS."); return

        for i, match_idx in enumerate(best_matches_indices_habis):
            best_match = database_df.iloc[match_idx]
            updates_habis.append({'range': f'R{i+2}C{sku_col_habis}', 'values': [[best_match['SKU']]]})
            updates_habis.append({'range': f'R{i+2}C{kategori_col_habis}', 'values': [[best_match['KATEGORI']]]})
            
    try:
        with st.spinner("Menulis pembaruan ke Google Sheets... Ini mungkin memakan waktu beberapa saat."):
            if updates_ready:
                db_klik_ready_sheet.batch_update(updates_ready)
            if updates_habis:
                db_klik_habis_sheet.batch_update(updates_habis)
        prog.progress(100)
        with placeholder.container(): st.success(f"Labeling Selesai! {len(db_klik_ready_df)} produk READY dan {len(db_klik_habis_df)} produk HABIS telah diperbarui.")
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

# --- Tombol utama untuk memuat data sekali di awal ---
if 'data_loaded' not in st.session_state:
    if st.button("Tarik Data & Mulai Analisis 🚀", type="primary"):
        df_data, db_df_data = load_all_data(SPREADSHEET_KEY)
        if df_data is not None:
            st.session_state.df = df_data
            st.session_state.db_df = db_df_data
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.error("Gagal memuat data utama.")
    st.stop()

df = st.session_state.df

# --- Sidebar Utama untuk Navigasi ---
st.sidebar.header("Menu Utama")
main_menu = st.sidebar.radio("Pilih Alat:", ("Dashboard Analisis", "Similarity Produk", "Tools (Peralatan)"))

if main_menu == "Dashboard Analisis":
    # ================================
    # BAGIAN DASHBOARD ANALISIS
    # ================================
    st.header("📈 Dashboard Analisis Penjualan & Kompetitor")
    
    st.sidebar.header("Filter Dashboard")
    min_date_val, max_date_val = df['Tanggal'].min().date(), df['Tanggal'].max().date()
    start_date, end_date = st.sidebar.date_input("Rentang Tanggal:", [min_date_val, max_date_val], min_value=min_date_val, max_value=max_date_val)
    if len((start_date, end_date)) != 2: st.sidebar.warning("Pilih 2 tanggal."); st.stop()
    
    start_date_dt, end_date_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df_filtered = df[(df['Tanggal'] >= start_date_dt) & (df['Tanggal'] <= end_date_dt)].copy()
    if df_filtered.empty: st.error("Tidak ada data di rentang tanggal yang dipilih."); st.stop()

    df_filtered['Minggu'] = df_filtered['Tanggal'].dt.to_period('W-SUN').apply(lambda p: p.start_time).dt.date
    latest_entries_overall = df_filtered.loc[df_filtered.groupby(['Toko', 'Nama Produk'])['Tanggal'].idxmax()]
    main_store_latest_overall = latest_entries_overall[latest_entries_overall['Toko'] == "DB KLIK"]
    competitor_latest_overall = latest_entries_overall[latest_entries_overall['Toko'] != "DB KLIK"]

    tab1, tab3, tab4, tab5, tab6 = st.tabs(["⭐ Toko Saya", "🏆 Brand Kompetitor", "📦 Status Stok", "📈 Kinerja Penjualan", "📊 Produk Baru"])
    
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

elif main_menu == "Similarity Produk":
    # ================================
    # BAGIAN ALAT SIMILARITY PRODUK
    # ================================
    st.header("⚖️ Alat Analisis Similarity Produk")
    st.info("Alat ini menggunakan TF-IDF dan Validasi Brand untuk menemukan produk serupa.")

    st.sidebar.header("Filter Similarity")
    accuracy_cutoff = st.sidebar.slider("Tingkat Akurasi Minimum (%)", 50, 100, 65, 1)

    df['Nama Normalisasi'] = df['Nama Produk'].apply(normalize_text)
    my_store_df = df[(df['Toko'] == "DB KLIK") & (df['Status'] == 'Tersedia')].copy()
    my_store_df.dropna(subset=['SKU', 'Brand'], inplace=True)
    my_store_df.drop_duplicates(subset='SKU', inplace=True)
    
    competitor_df = df[df['Toko'] != "DB KLIK"].copy()
    product_list = sorted(my_store_df['Nama Produk'].unique().tolist())

    if not product_list: st.warning("Tidak ada produk 'DB KLIK' yang tersedia untuk dianalisis."); st.stop()
    selected_name = st.selectbox("Pilih Produk Anda (DB KLIK):", product_list)

    if st.button("🔍 Analisis Produk", type="primary"):
        if selected_name:
            selected_sku = my_store_df[my_store_df['Nama Produk'] == selected_name].iloc[0]['SKU']
            with st.spinner("Menganalisis kemiripan produk..."):
                matches = find_matches_for_similarity_tool(selected_sku, my_store_df, competitor_df, score_cutoff=accuracy_cutoff/100)
            
            st.subheader("Hasil Analisis")
            if len(matches) > 1:
                display_df = pd.DataFrame(matches).sort_values(by='Skor Kemiripan (%)', ascending=False)
                my_price = display_df.iloc[0]['Harga']
                display_df['Selisih'] = display_df['Harga'] - my_price
                def format_selisih(val):
                    if val == 0: return "Rp 0 (Basis)"
                    status = " (Lebih Mahal)" if val > 0 else " (Lebih Murah)"
                    return f"Rp {val:,.0f}{status}"
                display_df['Harga'] = display_df['Harga'].apply(format_rupiah)
                display_df['Selisih Harga'] = display_df['Selisih'].apply(format_selisih)
                st.dataframe(display_df[['Nama Produk Tercantum', 'Toko', 'Harga', 'Selisih Harga', 'Status Stok', 'Skor Kemiripan (%)']], use_container_width=True, hide_index=True)
            else:
                st.warning("Tidak ditemukan produk yang cocok.")

elif main_menu == "Tools (Peralatan)":
    # ================================
    # BAGIAN ALAT LABELING
    # ================================
    st.header("🛠️ Tools (Peralatan)")
    st.subheader("Labeling SKU dan Kategori Otomatis")
    st.warning("PERHATIAN: Proses ini akan **menimpa (rewrite)** data SKU dan Kategori pada sheet `DB KLIK - REKAP - READY` dan `DB KLIK - REKAP - HABIS` secara permanen. Gunakan dengan hati-hati.")
    if st.button("🚀 Mulai Proses Labeling", type="primary"):
        run_sku_category_labeling(gc, SPREADSHEET_KEY)

