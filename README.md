# Laporan Proyek Machine Learning - Umar Tilmisani

## Non-Performing Loan Prediction Project

**Latar Belakang (Domain Proyek)**

Pada zaman ini, industri keuangan dan perbankan adalah salah satu sektor yang paling terdigitalisasi dan mengandalkan pengambilan keputusan berbasis data. Salah satu produk utama lembaga keuangan adalah pembiayaan kredit/pinjaman. Namun, keputusan pemberian pinjaman merupakan proses yang kompleks karena harus mempertimbangkan risiko gagal bayar dari calon peminjam.

Menurut data dari Otoritas Jasa Keuangan (OJK), tingkat kredit bermasalah atau Non-Performing Loan (NPL) perbankan di Indonesia pada 2023 tercatat berada di angka sekitar 2,5%. Untuk menekan angka ini, perbankan dan fintech mulai mengadopsi teknologi machine learning (ML) sebagai solusi dalam melakukan penilaian kelayakan kredit secara otomatis.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding
 
### Problem Statements

- Bagaimana cara menentukan apakah suatu pengajuan pinjaman dapat disetujui atau ditolak berdasarkan data historis calon nasabah yang mengajukan pinjaman?
- Faktor apa saja yang dapat mempengaruhi keputusan dalam memprediksi kelayakan calon nasabah?
- Model *machine learning* apa yang paling akurat untuk memprediksi kelayakan calon nasabah?

Bank sering kali mengalami kerugian akibat memberikan pinjaman kepada nasabah yang sebenarnya berisiko tinggi untuk gagal membayar. Dibutuhkan sistem yang dapat membantu memprediksi kelayakan pinjaman nasabah berdasarkan data historis seorang nasabah.

### Goals

- Membangun model *machine learning* klasifikasi yang dapat menentukan apakah pengajuan pinjaman disetujui atau tidak berdasarkan data historis calon nasabah.
- Menidentifikasi dan menganalisis faktor yang mempengaruhi dalam proses prediksi kelayakan calon nasabah.
- Membandingkan performa dari beberapa model *machine learning* dan memilih model terbaik dalam proses prediksi kelayakan calon nasabah.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements
- Dengan membangun model machine learning klasifikasi menggunakan beberapa algoritma seperti Random Forest, Logistic Regression, atau XGBoost yang dilatih menggunakan data historis pengajuan pinjaman.
- Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.
  
**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
*Dataset* yang digunakan pada proyek *machine learning* ini adalah **"*Loan Approval Prediction Dataset*"**, yang tersedia di *platform* [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset). *Dataset* ini berisi kumpulan data kuantitatif yang terdiri dari berbagai kolom yang menjadi faktor apakah pengajuan pinjaman ditolak atau disetujui. Secara keseluruhan, *dataset* ini terdiri dari 4269 baris dan 12 kolom.

*Dataset* ini sesuai dengan kebutuhan dalam membangun proyek machine learning, khususnya untuk tugas *binary classification*. 

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada *Loan Approval Prediction Dataset* adalah sebagai berikut:
- ```loan_id```: id atau idex dari data pengajuan pinjaman.
- ```no_of_dependents```: jumlah orang yang menjadi tanggungan pemohon atau nasabah.
- ```education```: status pendidikan pemohon atau nasabah.
- ```self_employed```: status pekerjaan pemohon apakah bekerja sendiri sebagai wirausaha atau bukan.
- ```income_annum```: nilai pendapatan tahunan pemohon.
- ```loan_ammount```: jumlah atau nominal pinjaman yang diajukan.
- ```loan_term```: waktu atau durasi pelunasan pinjaman dalam satuan bulan.
- ```cibil_score```: jumlah skor kredit pemohon atau nasabah.
- ```residential_assets_value```: nilai aset pemohon dalam bentuk rumah atau tempat tinggal.
- ```commercial_assets_value```: nilai aset komersial yang dimiliki oleh pemohon atau nasabah.
- ```luxury_assets_value```: nilai aset pemohon dalam bentuk barang mewah atau berharga.
- ```bank_assets_value```: nilai tabungan atau dana finansial yang dimiliki oleh pemohon atau nasabah.
- ```loan_status```: status pinjaman apakah disetujui atau ditolak.

Selanjutnya, untuk meingkatkan pemahaman mengenai data yang digunakan, penulis melakukan proses *exploratory data analysist* (EDA).

***Exploratory Data Analysist* (EDA)**

Exploratory Data Analysis (EDA) adalah pendekatan analisis data yang bertujuan untuk memahami karakteristik utama dari kumpulan data. EDA melibatkan penggunaan teknik statistik dan visualisasi grafis untuk menemukan pola, hubungan, atau anomali untuk membentuk hipotesis. Proses ini sering kali tidak terstruktur dan dianggap sebagai langkah awal penting dalam analisis data yang membantu menentukan arah analisis lebih lanjut.

Berikut tahapan EDA yang dilakukan:
<!-- ```py
df.info()
```
Output: -->
- Melihat informasi tabel pada *dataset*
    ```py
    RangeIndex: 4269 entries, 0 to 4268
    Data columns (total 13 columns):
    #   Column                     Non-Null Count  Dtype 
    ---  ------                     --------------  ----- 
    0   loan_id                    4269 non-null   int64 
    1    no_of_dependents          4269 non-null   int64 
    2    education                 4269 non-null   object
    3    self_employed             4269 non-null   object
    4    income_annum              4269 non-null   int64 
    5    loan_amount               4269 non-null   int64 
    6    loan_term                 4269 non-null   int64 
    7    cibil_score               4269 non-null   int64 
    8    residential_assets_value  4269 non-null   int64 
    9    commercial_assets_value   4269 non-null   int64 
    10   luxury_assets_value       4269 non-null   int64 
    11   bank_asset_value          4269 non-null   int64 
    12   loan_status               4269 non-null   object
    dtypes: int64(10), object(3)
    memory usage: 433.7+ KB
    ```
- Melihat analsis
- Memeriksa data *missing value* dan *duplicated data* 

**Visualisasi Data**
- *Univariate Data Analysist*
- *Multivariate Data Analysist*
- *Box Plots*
## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

