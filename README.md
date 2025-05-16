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


### Solution statements
- Dengan membangun model machine learning klasifikasi menggunakan beberapa algoritma seperti Random Forest, Logistic Regression, atau XGBoost yang dilatih menggunakan data historis pengajuan pinjaman.
- Melakukan proses *data preprocessing* agar kualitas data yang digunakan menjadi lebih baik sehingga model machine learning yang dibangun memiliki performa yang baik.
  
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
- ```loan_term```: jangka waktu atau durasi pinjaman dalam satuan tahun.
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
- **Melihat informasi tabel pada *dataset***
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
    **Insights**
    
- **Melihat ringkasan statistik deskriptif kolom numerik**
  
    | **Feature**                   | **count**   | **mean**        | **std**         | **min**     | *25%*      | **50%**      | **75%**       | **max**       |
    |---------------------------|---------|-------------|-------------|---------|----------|----------|-----------|-----------|
    | no_of_dependents          | 4269.0  | 2.498712e+00| 1.695910e+00| 0.0     | 1.0      | 3.0      | 4.0       | 5.0       |
    | income_annum              | 4269.0  | 5.059124e+06| 2.806840e+06| 200000.0| 2700000.0| 5100000.0| 7500000.0 | 9900000.0 |
    | loan_amount               | 4269.0  | 1.513345e+07| 9.043363e+06| 300000.0| 7700000.0| 14500000.0| 21500000.0| 39500000.0|
    | loan_term                 | 4269.0  | 1.090045e+01| 5.709187e+00| 2.0     | 6.0      | 10.0     | 16.0      | 20.0      |
    | cibil_score               | 4269.0  | 5.999361e+02| 1.724304e+02| 300.0   | 453.0    | 600.0    | 748.0     | 900.0     |
    | residential_assets_value  | 4269.0  | 7.472617e+06| 6.503637e+06| -100000.0|2200000.0 | 5600000.0| 11300000.0| 29100000.0|
    | commercial_assets_value   | 4269.0  | 4.973155e+06| 4.388986e+06| 0.0     | 1300000.0| 3700000.0| 7600000.0 | 19400000.0|
    | luxury_assets_value       | 4269.0  | 1.512631e+07| 9.103754e+06| 300000.0| 7500000.0| 14600000.0| 21700000.0| 39200000.0|
    | bank_asset_value          | 4269.0  | 4.976692e+06| 3.250185e+06| 0.0     | 2300000.0| 4600000.0| 7100000.0 | 14700000.0|

    **Insights**
    - ```no_of_depents```: Rata-rata tanggungan nasabah sebesar 2,5, dengan nilai maksimum 5 dan minimum 0. Mayoritas pemohon memiliki 1-4 orang tanggungan, data ini memengaruhi kelayakan pemohon untuk diberi pinjaman.
    - ```income_annum```: Rata-rata pendapatan tahunan 5 juta dengan nilai maksimum 9,9 juta dan minimum 200 ribu. rentang nilai pendapatan setiap pemohon sangat besar, perlu dilakukan normalisasi atau *scaling* pada proses *data preprocessing* sebelum dilakukan tahap *modeling*
    - ```loan_amount```: Rata-rata nilai pinjaman 15 juta dengan nilai maksimum 39,5 juta dan nilai minimum 300 ribu. Rentang nilai cukup tinggi berpotensi memiliki *outlier*.
    - ```loan_term```: Rata-rata 10 dengan nilai maksimum 20 dan nilai minimum 2. Mayoritas pemohon mengajukan pinjaman jangka panjang dengan waktu pinjaman cukup lama.
    - ```cibil_score```: Rata-rata nilai sekitar 600 dengan nilai maksimum 900 dan minimum 300.
    - ```residential_assets_value```: Rata-rata nilai aset residensial pemohon sekitar 7jt dengan nilai maksimum 29 juta dan nilai minimum -100 ribu. Data memiliki rentang nilai yang cukup jauh dan terdapat anomali yaitu terdapat nilai negatif. Nilai tersebut bisa jadi hasil dari kesalahan input data atau nilai aset berupa hutang. 
    - ```commercial_assets_value```: 
    - ```luxury_assets_value```: 
    - ```bank_assets_value```: 
 
- **Memeriksa data *missing value* dan *duplicated data***

**Visualisasi Data**
- ***Univariate Data Analysis***
  ![Univariate_Cat](https://github.com/user-attachments/assets/a2942633-255c-4fe7-aed9-e96396942de9)
  <div align="center">Gambar 1.1 - Univariate Analysis Categorical Column</div>
- ***Multivariate Data Analysis***
  ![Univariate_Num](https://github.com/user-attachments/assets/8bed92ff-86d1-491b-b69f-92ff574d21a9)
  <div align="center">Gambar 1.2 - Univariate Analysis Numerical Column</div>
- ***Box Plots***
  ![Image](https://github.com/user-attachments/assets/723edf9b-8f1d-403a-aef8-05fbe45f3421)
  <div align="center">Gambar 1.3 - </div>
  
  **Insights**
  Visualisasi data menggunakan *box plot* bertujuan untuk melihat distribusi data pada kolom numerik,  mengidentifikasi perbedaan distribusi antar kelas, serta mendeteksi keberadaan outliers yang dapat memengaruhi performa model. Dengan melihat median, rentang interkuartil (IQR), dan pencilan, boxplot membantu menentukan apakah fitur tertentu memiliki pengaruh signifikan terhadap target dan memberikan wawasan awal untuk pemilihan fitur atau penanganan data sebelum pemodelan.

- ***Pairplot of Numerical Features***
  ![Pairplot](https://github.com/user-attachments/assets/27c43ffb-2f2d-40d4-9199-8b85d2bb98a9)
<div align="center">Gambar 1.4 - Pairplot</div>

- Heatmap
![Heatmap](https://github.com/user-attachments/assets/868592c0-270f-4bcd-80d8-c22afeda717d)
<div align="center">Gambar 1.4 - Pairplot</div>

![Heatmap1](https://github.com/user-attachments/assets/669b760f-f7bb-4e4a-9392-3aace93036ae)
<div align="center">Gambar 1.4 - Pairplot</div>
  
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

