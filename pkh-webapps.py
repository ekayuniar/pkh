import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.naive_bayes import GaussianNB

st.write("""
# Klasifikasi Penerima Bantuan PKH (Web Apps)
Aplikasi berbasi Web untuk memprediski (mengklasifikasi) penerima bantuan PKH bagi masyarakat Desa "A" \n
""")

img = Image.open('gov_sas.jpg')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

st.sidebar.header('Keterangan Inputan Data')
st.sidebar.write("""1. Usia \n
    Lansia (46->65 tahun)  = 4 \n
    Anak-anak (5-11 tahun) = 3 \n
    Remaja (12-25 tahun)   = 2 \n
    Dewasa (26-45 tahun)   = 1 \n\n
    
2. Jumlah Tanggungan\n
    <2  = 3 \n
    <= 2 = 2 \n
    > 2  = 1  \n\n
    
3. Pekerjaan \n
    Pelajar/Mahasiswa     = 7 \n
    Mengurus Rumah Tangga = 6 \n
    Buruh Lepas           = 5 \n
    Buruh Tani            = 4 \n
    Petani/Pekebun        = 3 \n
    Karyawan/Wiraswasta   = 2 \n
    Wirausaha             = 1 \n
    PNS/TNI/Polri         = 0 \n\n
    
4. Penghasilan \n
    Tidak Berpenghasilan  = 4 \n
    < 1.000.000           = 3 \n
    1.000.000 – 2.000.000 = 2 \n
    > 2.000.000           = 1 \n\n
    
5. Tempat Tinggal \n
    Kontrak       = 3 \n
    Bebas Sewa    = 2 \n
    Milik Sendiri = 1 \n\n
    
6. Jenis Lantai  \n
    Tanah   = 3 \n
    Semen   = 2 \n
    Keramik = 1 \n\n
    
7. Jenis Dinding \n
    Anyaman Bambu = 3 \n
    Kayu          = 2 \n
    Semen         = 1 \n\n
    
8. Bantuan Lain \n
    Tidak = 2 \n
    Ya = 1
""")

modelnb = pickle.load(open('./Model/modelNBC_PKHv2.pkl', 'rb'))


def run():
    u = {4: 'LANSIA (46 - >65 TAHUN)', 3: 'ANAK-ANAK (5 - 11 TAHUN)',
         2: 'REMAJA (12 - 25 TAHUN)', 1: 'DEWASA (26 - 45 TAHUN)'}
    us = list(u.keys())
    usia = st.selectbox('USIA', us, format_func=lambda x: u[x])

    j = {3: '< 2', 2: '<=2', 1: '> 2'}
    jm = list(j.keys())
    jml_tgg = st.selectbox('JUMLAH_TANGGUNGAN', jm, format_func=lambda x: j[x])

    p = {7: 'PELAJAR/MAHASISWA', 6: 'MENGURUS RUMAH TANGGA', 5: 'BURUH LEPAS', 4: 'BURUH TANI',
         3: 'PETANI/PEKEBUN', 2: 'KARYAWAN SWASTA', 1: 'WIRASWASTA', 0: 'GURU/PNS/TNI/POLRI'}
    pk = list(p.keys())
    pekerjaan = st.selectbox('PEKERJAAN', pk, format_func=lambda x: p[x])

    ph = {4: 'TIDAK BERPENGHASILAN', 3: '< 1.000.000',
          2: '1.000.000 - 2.000.000', 1: '> 2.000.000'}
    phs = list(ph.keys())
    penghasilan = st.selectbox('PENGHASILAN', phs, format_func=lambda x: ph[x])

    t = {3: 'KONTRAK', 2: 'BEBAS SEWA', 1: 'MILIK SENDIRI'}
    tg = list(t.keys())
    tggl = st.selectbox('TEMPAT_TINGGAL', tg, format_func=lambda x: t[x])

    jl = {3: 'TANAH', 2: 'SEMEN', 1: 'KERAMIK'}
    jlt = list(jl.keys())
    jn_lt = st.selectbox('JENIS_LANTAI', jlt, format_func=lambda x: jl[x])

    jd = {3: 'ANYAMAN BAMBU', 2: 'KAYU', 1: 'SEMEN'}
    jdd = list(jd.keys())
    jn_dd = st.selectbox('JENIS_DINDING', jdd, format_func=lambda x: jd[x])

    b = {2: 'TIDAK', 1: 'YA'}
    bl = list(b.keys())
    bntl = st.selectbox('BANTUAN_LAIN', bl, format_func=lambda x: b[x])

    st.subheader('Tabel Inputan Data')
    data = {'USIA': usia,
            'JUMLAH_TANGGUNGAN': jml_tgg,
            'PEKERJAAN': pekerjaan,
            'PENGHASILAN': penghasilan,
            'TEMPAT_TINGGAL': tggl,
            'JENIS_LANTAI': jn_lt,
            'JENIS_DINDING': jn_dd,
            'BANTUAN_LAIN': bntl, }
    fitur = pd.DataFrame(data, index=[0])
    st.write(fitur)

    prediksi = modelnb.predict(fitur)
    pred_prob = modelnb.predict_proba(fitur)

    st.subheader('Keterangan Label Kelas')
    keterangan = np.array(['LAYAK', 'TIDAK LAYAK'])
    st.write(keterangan)

    st.subheader('Hasil Prediksi (Klasifikasi Penerima Bantuan PKH)')
    keterangan = np.array(0)
    st.write(prediksi[keterangan])

    st.subheader(
        'Probabilitas Hasil Prediksi (Klasifikasi Penerima Bantuan PKH)')
    st.write(pred_prob)


run()
