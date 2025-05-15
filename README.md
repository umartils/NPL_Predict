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
- Model machine learning apa yang paling akurat untuk memprediksi kelayakan calon nasabah?

Bank sering kali mengalami kerugian akibat memberikan pinjaman kepada nasabah yang sebenarnya berisiko tinggi untuk gagal membayar. Dibutuhkan sistem yang dapat membantu memprediksi kelayakan pinjaman nasabah berdasarkan data historis seorang nasabah.

### Goals

- Membangun model machine learning klasifikasi yang dapat menentukan apakah pengajuan pinjaman disetujui atau tidak berdasarkan data historis calon nasabah.
- Menidentifikasi dan menganalisis faktor yang mempengaruhi dalam proses prediksi kelayakan calon nasabah.
- Membandingkan performa dari beberapa model machine learning dan memilih model terbaik dalam proses prediksi kelayakan calon nasabah.

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
Dataset yang digunakan pada proyek *machine learning* ini adalah **"Loan Prediction Problem Dataset"** 
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

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

