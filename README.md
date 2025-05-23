# Laporan Proyek Machine Learning - Umar Tilmisani

## Non-Performing Loan Prediction Project

**Latar Belakang (Domain Proyek)**

Pada zaman ini, industri keuangan dan perbankan adalah salah satu sektor yang paling terdigitalisasi dan mengandalkan pengambilan keputusan berbasis data. Salah satu produk utama lembaga keuangan adalah pembiayaan kredit/pinjaman. Namun, keputusan pemberian pinjaman merupakan proses yang kompleks karena harus mempertimbangkan risiko gagal bayar dari calon peminjam. Karena dalam proses pemberian kredit atau pinjaman, bank atau lembaga keuangan akan selalu berhadapan dengan masalah kredit macet atau Non Performing Loan (NPL). Ratio NPL akan selalu berdampak terhadap keuntungan sebuah lembaga keuangan. Serta semakin tinggi ratio NPL maka akan semakin buruk pula kualitas kredit
pada bank atau lembaga keuangan tersebut [1].

Sejak 2013 hingga 2023 ratio NPL di Indonesia mengalami tren kenaikan, tercatat pada 2021 ratio NPL Indonesia sebesar 2.65% dan mendapat peringkat 4 tertinggi di ASEAN dan peringkat ke 7 dari 20 negara yang tergabung pada G20 [2]. Menurut data dari Otoritas Jasa Keuangan (OJK), tingkat kredit bermasalah atau Non-Performing Loan (NPL) perbankan di Indonesia pada 2023 tercatat berada di angka sekitar 2,5%. Untuk menekan angka ini, perbankan dan fintech mulai mengadopsi teknologi *machine learning* (ML) sebagai solusi dalam melakukan penilaian kelayakan kredit secara otomatis.


## Business Understanding
 
### Problem Statements

- Bagaimana cara menentukan apakah suatu pengajuan pinjaman dapat disetujui atau ditolak berdasarkan data historis calon nasabah yang mengajukan pinjaman?
- Faktor apa saja yang dapat mempengaruhi keputusan dalam memprediksi kelayakan calon nasabah?
- Model *machine learning* apa yang paling akurat untuk memprediksi kelayakan calon nasabah?

Bank sering kali mengalami kerugian akibat memberikan pinjaman kepada nasabah yang sebenarnya berisiko tinggi untuk gagal membayar. Dibutuhkan sistem yang dapat membantu memprediksi kelayakan pinjaman nasabah berdasarkan data historis seorang nasabah.

### Goals

- Membangun model *machine learning* klasifikasi yang dapat menentukan apakah pengajuan pinjaman disetujui atau tidak berdasarkan data historis calon nasabah.
- Mengidentifikasi dan menganalisis faktor yang mempengaruhi dalam proses prediksi kelayakan calon nasabah.
- Membandingkan performa dari beberapa model *machine learning* dan memilih model terbaik dalam proses prediksi kelayakan calon nasabah.


### Solution statements
- Melakukan proses *data preprocessing* agar kualitas data yang digunakan menjadi lebih baik sehingga model machine learning yang dibangun memiliki performa yang baik.
- Membangun model *machine learning* klasifikasi menggunakan beberapa algoritma seperti *Support Vector Machine*, *Random Forest*, dan XGBoost yang dilatih menggunakan data historis pengajuan pinjaman.
- Menggunakan metrik evaluasi seperti *accuracy*, *precision*, *recall*, *f1-score*, *confusion matrix*, dan roc auc untuk mengukur performa model dalam melakukan klasifikasi biner.


## Data Understanding
*Dataset* yang digunakan pada proyek *machine learning* ini adalah **"*Loan Approval Prediction Dataset*"**, yang tersedia di *platform* [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset). *Dataset* ini berisi kumpulan data kuantitatif yang terdiri dari berbagai kolom yang menjadi faktor apakah pengajuan pinjaman ditolak atau disetujui. Secara keseluruhan, *dataset* ini terdiri dari 4269 baris dan 13 kolom. *Dataset* ini sesuai dengan kebutuhan dalam membangun proyek machine learning, khususnya untuk tugas *binary classification*. 

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
    **Insight**
    - ```loan_id```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```no_of_dependents```: jmemiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```education```: memiliki tipe data ```object``` atau data kategorikal dengan jumlah data sebanyak 4269..
  - ```self_employed```: memiliki tipe data ```object``` atau data kategorikal dengan jumlah data sebanyak 4269..
  - ```income_annum```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```loan_ammount```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```loan_term```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```cibil_score```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```residential_assets_value```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```commercial_assets_value```: nmemiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```luxury_assets_value```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```bank_assets_value```: memiliki tipe data ```int64``` dengan jumlah data sebanyak 4269.
  - ```loan_status```: memiliki tipe data ```object``` atau data kategorikal dengan jumlah data sebanyak 4269.
  
  <br>
    
- **Melihat ringkasan statistik deskriptif kolom numerik**
  
    |                    | count   | mean        | std         | min     | *25%*      | 50%      | 75%       | max       |
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

    **Insight**

    - ```no_of_depents```: Rata-rata tanggungan nasabah sebesar 2,5, dengan nilai maksimum 5 dan minimum 0. Mayoritas pemohon memiliki 1-4 orang tanggungan, data ini memengaruhi kelayakan pemohon untuk diberi pinjaman.
    - ```income_annum```: Rata-rata pendapatan tahunan 5 juta dengan nilai maksimum 9,9 juta dan minimum 200 ribu. rentang nilai pendapatan setiap pemohon sangat besar, perlu dilakukan normalisasi atau *scaling* pada proses *data preprocessing* sebelum dilakukan tahap *modeling*
    - ```loan_amount```: Rata-rata nilai pinjaman 15 juta dengan nilai maksimum 39,5 juta dan nilai minimum 300 ribu. Rentang nilai cukup tinggi berpotensi memiliki *outlier*.
    - ```loan_term```: Rata-rata 10 dengan nilai maksimum 20 dan nilai minimum 2. Mayoritas pemohon mengajukan pinjaman jangka panjang dengan waktu pinjaman cukup lama.
    - ```cibil_score```: Rata-rata nilai sekitar 600 dengan nilai maksimum 900 dan minimum 300.
    - ```residential_assets_value```: Rata-rata nilai aset residensial pemohon sekitar 7jt dengan nilai maksimum 29 juta dan nilai minimum -100 ribu. Data memiliki rentang nilai yang cukup jauh dan terdapat anomali yaitu terdapat nilai negatif. Nilai tersebut bisa jadi hasil dari kesalahan input data atau nilai aset berupa hutang. 
    - ```commercial_assets_value```: Rata-rata nilai aset komersial sebesar 4jt dengan nilai maksimum sebesar 19,4 juta dan nilai minimum 0. Berpotensi memiliki outlier karena memiliki range nilai yang cukup besar.
    - ```luxury_assets_value```: Rata-rata nilai aset barang berharga sebesar 15 juta dengan nilai maksimum 39,2 juta dan nilai minimum sebesar 300 ribu. Berpotensi memiliki outlier karena memiliki range nilai yang cukup besar.
    - ```bank_assets_value```: Rata-rata nilai aset di bank sebesar 4 juta dengan nilai maksimum 14,7 juta dan nilai minimum 0.
  
  <br>
 
- **Memeriksa data *missing value* dan *duplicated data***

    - Tidak ada data yang hilang karena semua kolom pada dataset memiliki jumlah baris yang sama dengan total jumlah data.
    - Tidak terdapat data duplikat, sehingga data sudah bisa digunakan tanpa harus menghilangkan data yang duplikat
  
  <br>

**Visualisasi Data**
- ***Univariate Data Analysis***
  Univariate Analysis merupakan metode analisis data yang berfokus pada pemeriksaan satu variabel atau kolom data secara individual. Tujuannya adalah untuk memberikan gambaran deskriptif mengenai data tersebut serta mengidentifikasi pola-pola yang terdapat dalam sebaran nilainya. Teknik yang umum digunakan meliputi statistik deskriptif, histogram, dan diagram kotak (box plot) untuk menganalisis distribusi dan memahami karakteristik variabel yang bersangkutan.

 
  ![Univariate_Cat](https://github.com/user-attachments/assets/a2942633-255c-4fe7-aed9-e96396942de9)
  <div align="center">Gambar 1 - Univariate Analysis Categorical Column</div>
  
  **Insight**

  Pada ```Gambar 1``` menampilkan distribusi nilai pada kolom kategorikal yang ada di *dataset*. Terlihat bahwa kolom ```education``` dan ```self_employed``` memiliki distribusi yang merata dari setiap kategorinya, sedangkan pada kolom ``loan_status`` distribusi datanya kurang merata dimana kategori `Approved` memiliki jumlah yang lebih banyak dibanding kategori ``Rejected``. Perlu dilakukan proses *feature engineering* agar jumlah data pada kedua kategori dapat seimbang sehingga hasil dari model *machine learning* memiliki performa yang lebih baik. <br>
  ![Univariate_Num](https://github.com/user-attachments/assets/8bed92ff-86d1-491b-b69f-92ff574d21a9)
  <div align="center">Gambar 2 - Univariate Analysis Numerical Column</div>
  
  **Insight**

  Berikut beberapa insight yang diperoleh dari ```Gambar 2``` mengenai distribusi nilai pada kolom numerik dalam dataset: 
  - ```no_of_dependents```: memiliki distribusi diskrit dan merata dengan rentang 0 hingga 5, artinya pemohon tersebar relatif seimbang berdasarkan jumlah tanggungan.
  - ```income_annum```: memiliki distribusi cenderung seragam (*uniform*) dengan sedikit variasi. 
  - ```loan_amount```: memiliki distribusi cenderung *right skewed* 
  - ```loan_term```: memiliki distribusi diskrit dan merata dengan rentang 2,5 hingga 20 tahun, artinya jangka waktu pinjaman tersebar merata.
  - ```cibil_score```: nilai skor kredit terdistribusi secara merata.
  - ```residential_assets_value```: memiliki distribusi *right skewed* dimana nilai mean lebih besar dibanding median dan mode.
  - ```commercial_assets_value```: memiliki distribusi *right skewed* dimana nilai mean lebih besar dibanding median dan mode.
  - ```luxury_assets_value```: memiliki distribusi *right skewed* dimana nilai mean lebih besar dibanding median dan mode.
  - ```bank_assets_value```: memiliki distribusi *right skewed* dimana nilai mean lebih besar dibanding median dan mode.
<br>

- ***Multivariate Data Analysis***
  ![Pairplot](https://github.com/user-attachments/assets/27c43ffb-2f2d-40d4-9199-8b85d2bb98a9)
  <div align="center">Gambar 3 - Pairplot</div>

  **Insight**

  Pada `Gambar 3` menampilkan hubungan multivariat antar kolom numerik terhadap kolom `loan_status` (Approved/Rejected). Berikut tambahan informasi yang diperoleh:

  * Kolom `residential_assets_value`, `commercial_assets_value`, dan `luxury_assets_value` memiliki distribusi yang cenderung tidak normal dan sebagian besar datanya terkonsentrasi di nilai rendah (left-skewed).
  * Hubungan linier juga terlihat antara:

    * `income_annum` dan `luxury_assets_value`
    * `income_annum` dan `bank_asset_value`
    * `loan_amount` dan `luxury_assets_value`
  * Warna titik-titik pada plot (Approved - biru, Rejected - oranye) memperlihatkan bahwa distribusi pemohon pinjaman yang disetujui lebih tinggi secara umum dibandingkan yang ditolak, khususnya pada nilai-nilai:

    * `cibil_score` tinggi
    * `income_annum` dan `loan_amount` menengah ke atas
  * Terdapat persebaran data yang cukup luas pada beberapa kolom seperti `income_annum` dan `loan_amount`, tetapi masih memperlihatkan pola yang dapat membantu dalam klasifikasi status pinjaman.

  **Kesimpulan**
  Visualisasi ini memperkuat temuan sebelumnya bahwa fitur seperti `cibil_score`, `income_annum`, `loan_amount`, dan `luxury_assets_value` sangat berpengaruh dalam menentukan status pinjaman. Distribusi serta hubungan linier antar fitur juga penting untuk dipertimbangkan saat membangun model prediktif.

  

- ***Box Plots***
 
  Visualisasi data menggunakan *box plot* bertujuan untuk melihat distribusi data pada kolom numerik,  mengidentifikasi perbedaan distribusi antar kelas, serta mendeteksi keberadaan outliers yang dapat memengaruhi performa model. Dengan melihat median, rentang interkuartil (IQR), dan pencilan, boxplot membantu menentukan apakah fitur tertentu memiliki pengaruh signifikan terhadap target dan memberikan wawasan awal untuk pemilihan fitur atau penanganan data sebelum pemodelan.<br>
  ![Image](https://github.com/user-attachments/assets/723edf9b-8f1d-403a-aef8-05fbe45f3421)
  <div align="center">Gambar 4 - Boxplot </div>
  
  **Insight**

  Berdasarkan ```Gambar 4``` terdapat kolom yang memiliki *outlier* yaitu kolom ```residential_assets_value```, ```commercial_assets_value```, dan ```bank_assets_value```. Kondisi ini terjadi apabila terdapat data yang bernilai ekstrem atau jauh dari nilai mayoritas pada data.
  <br>
  
- ***Heatmap* Korelasi Setiap Fitur Numerik**
![Heatmap1](https://github.com/user-attachments/assets/669b760f-f7bb-4e4a-9392-3aace93036ae)<div align="center">Gambar 5 - Heatmap Korelasi Setiap Fitur Numerik</div>

  **Insight**

  Pada `Gambar 5` menampilkan korelasi setiap kolom numerik pada data. Berikut informasi yang didapatkan dari gambar tersebut:

  * `income_annum` dan `loan_amount` memiliki korelasi yang paling tinggi yaitu sebesar **0.93** atau **93%**
  * `income_annum` dan `luxury_assets_value` juga memiliki korelasi tinggi yaitu sebesar **0.93** atau **93%**
  * `loan_amount` berkorelasi tinggi dengan `luxury_assets_value` sebesar **0.86** atau **86%**
  * `bank_asset_value` memiliki korelasi kuat terhadap `income_annum` (**0.85**) dan `loan_amount` (**0.79**)
  * Korelasi paling signifikan terhadap variabel target `loan_status` berasal dari `cibil_score`, dengan nilai korelasi sebesar **0.77**, yang menunjukkan bahwa skor kredit sangat berpengaruh terhadap status pinjaman
  * Sebagian besar kolom lainnya memiliki korelasi yang rendah atau hampir tidak berkorelasi dengan `loan_status`, seperti `loan_term` (-0.11), `no_of_dependents` (-0.02), dan `residential_assets_value` (-0.01)

  **Kesimpulan**
  Beberapa fitur seperti `cibil_score`, `income_annum`, `loan_amount`, dan `luxury_assets_value` dapat menjadi fitur penting dalam prediksi status pinjaman karena memiliki korelasi yang cukup tinggi terhadap `loan_status` ataupun antar fitur yang relevan.


- ***Heatmap* Korelasi Setiap Fitur Numerik dengan Label**
![Heatmap](https://github.com/user-attachments/assets/868592c0-270f-4bcd-80d8-c22afeda717d)<div align="center">Gambar 6 - Heatmap Korelasi Pearson</div>

  **Insight**

  Pada `Gambar 6` menunjukkan korelasi Pearson antara setiap kolom numerik terhadap kolom label `loan_status`. Terlihat bahwa:

  * Fitur dengan nilai korelasi tertinggi terhadap kolom `loan_status` adalah **`cibil_score`** dengan nilai korelasi sebesar **0.77**, yang berarti hubungan positif yang kuat. Artinya, semakin tinggi nilai `cibil_score`, maka semakin besar kemungkinan pinjaman akan **disetujui (*Approved*)**.
  * Sementara itu, fitur lain seperti:

    * `loan_amount` (0.02),
    * `income_annum` (-0.02),
    * `residential_assets_value` (-0.01),
    * `bank_asset_value` (-0.01),
    * dan lainnya, menunjukkan korelasi yang sangat lemah atau hampir tidak ada hubungan dengan status pinjaman.
  * Korelasi negatif paling besar ditunjukkan oleh `loan_term` (-0.11), yang mengindikasikan bahwa semakin panjang jangka waktu pinjaman, kemungkinan untuk disetujui sedikit lebih rendah, meskipun hubungan ini juga masih sangat lemah.


  **Kesimpulan**

  * Dari hasil ini, dapat disimpulkan bahwa **`cibil_score`** merupakan fitur paling berpengaruh dalam menentukan status pinjaman.
  * Fitur numerik lain mungkin tidak secara langsung memengaruhi keputusan, atau kontribusinya baru terlihat dalam interaksi dengan fitur lain (non-linear relationship).
  * Oleh karena itu, penting untuk mempertimbangkan fitur tambahan atau transformasi fitur (feature engineering) serta mencoba model non-linear untuk mengeksplorasi hubungan yang lebih kompleks dalam prediksi `loan_status`.
  <br>







## Data Preparation
Data Preparation adalah proses pembersihan, transformasi, dan pengorganisasian data mentah ke dalam format yang dapat dipahami oleh algoritma pembelajaran mesin. Berikut ini adalah urutan langkah-langkah Data Preparation yang dilakukan beserta penjelasan dan alasannya:

- ***Data Cleaning***
  
  Pada tahap ini, data dibersihkan untuk meningkatkan kualitas dari data yang akan digunakan untuk melakukan pemodelan. Pada tahap ini ada beberapa hal yang akan dilakukan, yaitu sebagai berikut:

  - Menghapus nilai tidak relevan
  
    Pada tahap ini, dilakukan proses pembersihan data dengan menghilangkan nilai yang tidak relevan. Pada dataset yang digunakan, terdapat nilai tidak relevan pada kolom `residential_assets_value` dimana terdapat data yang bernilai negatif. Setelah diperiksa ternyata semua data yang bernilai negatif memiliki nilai yang sama, yaitu -100000.

    **Data Sebelum Dibersihkan**
    
    |                    | count   | mean        | std         | min     | *25%*      | 50%      | 75%       | max       |
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

    Untuk membersihkan nilai tidak relevan, saya menjalan kode sebagai berikut.

    ```py
    data = df.drop(columns='loan_id')
    data = data[data['residential_assets_value'] > 0]
    data.describe().T 
    ```

    **output**

    |                     | count  | mean        | std         | min      | 25%       | 50%       | 75%        | max       |
    |---------------------------|--------|-------------|-------------|----------|-----------|-----------|------------|-----------|
    | no_of_dependents          | 4241.0 | 2.497996e+00 | 1.695599e+00 | 0.0      | 1.0       | 3.0       | 4.0        | 5.0       |
    | income_annum              | 4241.0 | 5.074251e+06 | 2.803166e+06 | 200000.0 | 2700000.0 | 5100000.0 | 7500000.0  | 9900000.0 |
    | loan_amount               | 4241.0 | 1.517840e+07 | 9.034490e+06 | 300000.0 | 7700000.0 | 14600000.0| 21500000.0 | 39500000.0|
    | loan_term                 | 4241.0 | 1.090215e+01 | 5.708988e+00 | 2.0      | 6.0       | 10.0      | 16.0       | 20.0      |
    | cibil_score               | 4241.0 | 5.996857e+02 | 1.722773e+02 | 300.0    | 453.0     | 600.0     | 747.0      | 900.0     |
    | residential_assets_value  | 4241.0 | 7.522613e+06 | 6.495800e+06 | 0.0      | 2200000.0 | 5700000.0 | 11400000.0 | 29100000.0|
    | commercial_assets_value   | 4241.0 | 4.985121e+06 | 4.391504e+06 | 0.0      | 1300000.0 | 3700000.0 | 7700000.0  | 19400000.0|
    | luxury_assets_value       | 4241.0 | 1.517121e+07 | 9.094717e+06 | 300000.0 | 7500000.0 | 14600000.0| 21700000.0 | 39200000.0|
    | bank_asset_value          | 4241.0 | 4.991488e+06 | 3.249494e+06 | 0.0      | 2400000.0 | 4600000.0 | 7100000.0  | 14700000.0|


    

  - Menangani *Outliers*

    *Outlier* adalah data yang memiliki nilai sangat jauh dari mayoritas data lainnya. Misalnya, jika kebanyakan gaji dalam dataset berada di kisaran 3 juta–10 juta, lalu ada satu data dengan gaji 200 juta, maka data tersebut kemungkinan merupakan *outlier*. Outlier dapat muncul karena kesalahan input, variasi alami, atau fenomena luar biasa. Pada tahap ini, nilai yang menjadi *outlier* akan dihilangkan agar persebaran data lebih merata dan meningkatkan kualitas data. Metode yang digunakan dalam menangani outlier adalah menggunakan metode *Interquarile Range* (IQR), dengan langkah-langkah berikut:
    
    - Hitung nilai kuartil pertama (Q1) dan kuartil ketiga (Q3) dari kolom data numerik.
    - Hitung nilai IQR = Q3 - Q1.
    - Tentukan batas atas dan batas bawah.
    - Hapus data *outlier* dari dataset.

    Berikut proses implementasi metode IQR untuk menangani *outlier*:

    - **Distribusi data sebelum implementasi IQR**

      ![Image](https://github.com/user-attachments/assets/723edf9b-8f1d-403a-aef8-05fbe45f3421)
      <div align="center">Gambar 7 - Distribusi data sebelum proses IQR </div>
    
    - **Implementasi metode IQR**
  
      ```py

      data = df.drop(columns="loand_id")
      numeric_columns = data.select_dtypes(include='int64').columns

      for kolom in numeric_columns:
        q1 = data[kolom].quantile(0.25)
        q3 = data[kolom].quantile(0.75)
        iqr = q3-q1

        low_bound = q1 - 1.5 * iqr
        up_bound = q3 + 1.5 * iqr
        data = data[(data[kolom] > low_bound) & (data[kolom] < up_bound)]
      ```
    - **Distribusi data setelah implementasi IQR**

      ![Image](https://github.com/user-attachments/assets/0939208c-82b1-4a7e-9819-678723c84e00)
      <div align="center">Gambar 8 - Distribusi data setelah proses IQR </div>
  
  
- ***Data Transformation***
  
  Tahap transformasi data merupakan tahap untuk mengubah bentuk atau format data mentah menjadi data yang siap untuk digunakan pemodelan. Proses transformasi data sangat penting dilakukan agar model yang dibangun memiliki performa yang baik. Berikut beberapa tahapan yang akan dilakukan dalam proses transformasi data:

  - **Enkode Data Kategorical (Data *Encoding*)**

    *Data encoding* merupakan proses mengubah data kategorikal ke dalam bentuk numerik. Hal ini dilakukan karena model tidak dapat membaca data dalam bentuk teks atau kategorikal dan hanya dapat membaca data berupa angka atau bentuk numerik, sehingga tahap ini sangat penting jika terdapat data kategorikal pada dataset. 

    Metode yang digunakan dalam proses *data encoding* adalah *label encoding*, dimana setiap kategori diubah ke dalam bentuk satu digit angka. Hal ini dilakukan karena setiap kolom yang memiliki data kategorikal pada dataset hanya terdapat dua jenis data saja, seperti *Graduate* atau *Not Graduate* pada kolom ```education```, lalu *yes* atau *no* pada kolom ```self_employed```, dan *Approved* atau *Rejected* pada kolom ```loan_status```. Berikut merupakan gambaran dari proses *data encoding*.

    
    **Kode**

    ```py
    categorical_columns = df.select_dtypes(include='object').columns
    print(df[categorical_columns].head())
    ```


    **Output**

    
    | Index | Education     | Self Employed | Loan Status |
    |-------|---------------|---------------|-------------|
    | 0     | Not Graduate  | Yes           | Rejected    |
    | 1     | Graduate      | No            | Rejected    |
    | 2     | Graduate      | No            | Rejected    |
    | 3     | Not Graduate  | Yes           | Rejected    |
    | 4     | Graduate      | Yes           | Rejected    |

    **Kode**

    ```py
    df['education'] = df['education'].map({' Graduate': 1, ' Not Graduate': 0})
    df['self_employed'] = df['self_employed'].map({' Yes': 1, ' No': 0})
    df['loan_status'] = df['loan_status'].map({' Approved': 1, ' Rejected': 0})
    print(df[cat_col].head())
    ```


    **Output**

    
    | Index | Education| Self Employed | Loan Status |
    |-------|----------|---------------|-------------|
    | 0     | 0        | 1             | 0           |
    | 1     | 1        | 0             | 0           |
    | 2     | 1        | 0             | 0           |
    | 3     | 0        | 1             | 0           |
    | 4     | 1        | 1             | 0           |

  - **Standarisasi Data**

    Standarisasi data merupakan proses mengubah nilai-nilai pada fitur numerik agar berada dalam skala tertentu tanpa mengubah distribusi relatif data. Proses ini perlu dilakukan terutama pada dengan rentang nilai yang sangat bervariasi antar fitur, karena beberapa model *machine learning* dapat terpengaruh oleh fitur dengan skala dominan.

    Standarasi data bertujuan agar data memiliki rentang yang sama, sehingga tidak ada fitur yang dominan karena memiliki nilai yang jauh lebih besar dibanding fitur lainnya. Selain itu juga dengan melakukan standarisai dapat meningkatkan performa model *machine learning* dibanding jika data tidak dalam rentang yang sama. Berikut penerapan dari proses standarisasi data.

    **Data Sebelum Standarisasi**
    

    |                     | count  | mean        | std         | min      | 25%       | 50%       | 75%        | max       |
    |---------------------------|--------|-------------|-------------|----------|-----------|-----------|------------|-----------|
    | no_of_dependents          | 4148.0 | 2.499759e+00 | 1.694329e+00 | 0.0      | 1.0       | 3.0       | 4.0        | 5.0       |
    | education                 | 4148.0 | 5.038573e-01 | 5.000454e-01 | 0.0      | 0.0       | 1.0       | 1.0        | 1.0       |
    | self_employed             | 4148.0 | 5.031340e-01 | 5.000505e-01 | 0.0      | 0.0       | 1.0       | 1.0        | 1.0       |
    | income_annum              | 4148.0 | 4.976230e+06 | 2.755451e+06 | 200000.0 | 2600000.0 | 5000000.0 | 7400000.0  | 9900000.0 |
    | loan_amount               | 4148.0 | 1.490921e+07 | 8.914531e+06 | 300000.0 | 7600000.0 | 14300000.0| 21100000.0 | 39500000.0|
    | loan_term                 | 4148.0 | 1.089007e+01 | 5.708494e+00 | 2.0      | 6.0       | 10.0      | 16.0       | 20.0      |
    | cibil_score               | 4148.0 | 5.999298e+02 | 1.721955e+02 | 300.0    | 453.75    | 600.0     | 747.0      | 900.0     |
    | residential_assets_value  | 4148.0 | 7.244455e+06 | 6.144507e+06 | 0.0      | 2200000.0 | 5500000.0 | 11000000.0 | 25100000.0|
    | commercial_assets_value   | 4148.0 | 4.834616e+06 | 4.210183e+06 | 0.0      | 1300000.0 | 3600000.0 | 7500000.0  | 17000000.0|
    | luxury_assets_value       | 4148.0 | 1.490137e+07 | 8.977879e+06 | 300000.0 | 7375000.0 | 14300000.0| 21225000.0 | 39200000.0|
    | bank_asset_value          | 4148.0 | 4.880231e+06 | 3.168089e+06 | 0.0      | 2300000.0 | 4500000.0 | 7000000.0  | 14000000.0|
    | loan_status               | 4148.0 | 6.231919e-01 | 4.846446e-01 | 0.0      | 0.0       | 1.0       | 1.0        | 1.0       |


    **Data Setelah Standarisasi**

    |                    | count  | mean     | std      | min | 25%      | 50%      | 75%      | max |
    |--------------------------|--------|----------|----------|-----|----------|----------|----------|-----|
    | no_of_dependents         | 4148.0 | 0.499952 | 0.338866 | 0.0 | 0.200000 | 0.600000 | 0.800000 | 1.0 |
    | income_annum             | 4148.0 | 0.492395 | 0.284067 | 0.0 | 0.247423 | 0.494845 | 0.742268 | 1.0 |
    | loan_amount              | 4148.0 | 0.372684 | 0.227412 | 0.0 | 0.186224 | 0.357143 | 0.530612 | 1.0 |
    | loan_term                | 4148.0 | 0.493893 | 0.317139 | 0.0 | 0.222222 | 0.444444 | 0.777778 | 1.0 |
    | cibil_score              | 4148.0 | 0.499883 | 0.286993 | 0.0 | 0.256250 | 0.500000 | 0.745000 | 1.0 |
    | residential_assets_value | 4148.0 | 0.288624 | 0.244801 | 0.0 | 0.087649 | 0.219124 | 0.438247 | 1.0 |
    | commercial_assets_value  | 4148.0 | 0.284362 | 0.247658 | 0.0 | 0.076471 | 0.211765 | 0.441176 | 1.0 |
    | luxury_assets_value      | 4148.0 | 0.375357 | 0.230794 | 0.0 | 0.181877 | 0.359897 | 0.537918 | 1.0 |
    | bank_asset_value         | 4148.0 | 0.348588 | 0.226292 | 0.0 | 0.164286 | 0.321429 | 0.500000 | 1.0 |


    Terlihat perbedaan nilai antara data sebelum standarisasi dan setelah standarisasi. Pada data sebelum standarisasi, rentang nilai setiap fitur sangat bervariasi, sedangkan setelah standarisasi rentang nilai semua fitur berada di rentang 0 hingga 1. Meskipun skala data telah diubah, distribusi relatif antar nilai dalam setiap fitur tetap terjaga. Artinya, pola hubungan antar data dalam setiap fitur tidak berubah, hanya dinormalisasi agar berada dalam skala yang sama. Selain itu, proses standarisasi tidak mengubah jumlah data atau strukturnya, hanya memodifikasi nilai numerik dari fitur numerik yang ada.

  - **Mengatasi *Imbalanced Data*** 

    *Imbalanced data* merupakan kondisi ketika distribusi kelas dalam dataset tidak seimbang, yaitu salah satu kelas (biasanya kelas mayoritas) memiliki jumlah data yang jauh lebih banyak dibandingkan kelas lainnya (kelas minoritas). Ketika distribusi kelas tidak seimbang, model *machine learning* yang dibangun cenderung bias terhadap kelas mayoritas dan mengabaikan kelas minoritas. Akibatnya, meskipun nilai akurasi model tampak tinggi, model tersebut dapat menipu karena gagal memprediksi kelas minoritas secara akurat. Hal ini menyebabkan performa model secara keseluruhan menjadi tidak optimal, khususnya jika metrik evaluasi seperti *precision*, *recall*, atau *F1-score* digunakan. Kondisi *imbalanced data* bisa dilihat seperti pada ```Gambar 7```.
  
    <p align="center">
    <img src="https://github.com/user-attachments/assets/c00f7072-d4cc-4118-8558-a971822fa07c" alt="imbalanced_data" />
    </p><div align="center">Gambar 9 - Distribusi Kelas Data Tidak Seimbang</div>
    
    Untuk mengatasi kondisi *imbalanced data*, terdapat beberapa pendekatan yang umum digunakan, antara lain:

    * ***Oversampling***: Menambahkan jumlah data pada kelas minoritas agar setara dengan kelas mayoritas.
    * ***Undersampling***: Mengurangi jumlah data pada kelas mayoritas agar seimbang dengan kelas minoritas.
    * ***Hybrid methods***: Menggabungkan *oversampling* dan undersampling.

    Pada tahap ini, teknik yang digunakan adalah ***oversampling***. Oversampling bertujuan untuk meningkatkan proporsi kelas minoritas dalam dataset agar distribusi kelas menjadi seimbang. Beberapa metode *oversampling* yang umum digunakan adalah:

    * ***Random Oversampling***: Duplikasi acak terhadap data dari kelas minoritas hingga jumlahnya setara dengan kelas mayoritas. Meskipun sederhana dan cepat, metode ini berisiko menimbulkan *overfitting* karena data yang sama diulang berkali-kali.

    * **SMOTE (*Synthetic Minority Over-sampling Technique*)**: Membuat data sintetis baru untuk kelas minoritas dengan cara menginterpolasi antar titik data terdekat. Teknik ini lebih canggih dibanding random *oversampling* karena menghasilkan variasi data baru yang dapat membantu model belajar pola dengan lebih baik.

    Dengan menerapkan teknik *oversampling*, distribusi kelas dalam dataset menjadi lebih seimbang seperti pada `Gambar 8`, sehingga model yang dibangun dapat belajar lebih adil terhadap seluruh kelas dan menghasilkan prediksi yang lebih akurat serta representatif, terutama dalam mengidentifikasi kelas minoritas.
 

    <p align="center">
    <img src="https://github.com/user-attachments/assets/c4b6697c-fc9e-4ee0-84b9-c819ed9db546" alt="imbalanced_data" />
    </p><div align="center">Gambar 10 - Distribusi Kelas Data Seimbang</div>
    

- ***Data Splitting***
  
  *Data splitting* adalah proses memisahkan dataset menjadi dua bagian utama, yaitu **data latih (*training set*)** dan **data uji (*testing set*)**. Tujuan dari proses ini adalah untuk mengevaluasi kinerja model secara objektif terhadap data yang belum pernah dilihat sebelumnya. Rasio yang umum digunakan dalam pemisahan data adalah **80:20**, artinya:

  * **80%** dari total data digunakan sebagai **data latih** untuk melatih model machine learning.
  * **20%** sisanya digunakan sebagai **data uji** untuk menguji performa model.

  Proses ini sangat penting agar model tidak hanya mengingat data pelatihan (*overfitting*), tetapi juga mampu melakukan generalisasi dengan baik terhadap data baru.

  **Tujuan Data Splitting (80:20):**

  - **Melatih model secara efektif:**
     Sebagian besar data (80%) digunakan untuk melatih model agar dapat memahami pola dan hubungan antar fitur serta target.

  - **Mengukur performa model secara objektif:**
     Dengan menyisihkan 20% data untuk pengujian, kita dapat mengevaluasi seberapa baik model bekerja pada data yang belum pernah dilihat sebelumnya (*unseen data*).

  - **Menghindari *overfitting*:**
     Dengan adanya data uji yang terpisah, kita bisa mengetahui apakah model hanya menghafal data pelatihan atau benar-benar mampu melakukan generalisasi.

  - **Menjaga efisiensi dan keadilan dalam evaluasi:**
     Rasio 80:20 memberikan keseimbangan antara cukupnya data untuk pelatihan dan tersedianya data yang cukup untuk evaluasi. Jika proporsi data uji terlalu kecil, evaluasi bisa menjadi bias; jika terlalu besar, data latih menjadi tidak cukup.



## Modeling

Tahapan modeling merupakan inti dari proyek machine learning yang bertujuan untuk membangun model prediktif berdasarkan data yang telah melalui proses pembersihan dan pra-pemrosesan. Dalam proyek ini, tugas utama machine learning adalah melakukan *prediksi biner*, yaitu menentukan apakah permohonan pinjaman dari seorang pemohon akan **disetujui** atau **ditolak**. Oleh karena itu, pendekatan klasifikasi biner digunakan sebagai solusi permasalahan.

Untuk mendapatkan performa terbaik, proyek ini membandingkan tiga algoritma klasifikasi populer yang memiliki karakteristik dan keunggulan masing-masing, yaitu:

1. **Support Vector Machine (SVM)**
   

2. **Random Forest Classifier**
   

3. **XGBoost (Extreme Gradient Boosting)**

Setiap model dilatih menggunakan data latih (*training set*) dan dievaluasi menggunakan data uji (*test set*) untuk mengukur kinerjanya dalam melakukan klasifikasi biner. Evaluasi dilakukan dengan metrik-metrik seperti akurasi, precision, recall, dan F1-score, untuk memberikan gambaran menyeluruh terhadap kemampuan prediktif dari masing-masing model.


### Support Vector Machine


S*upport Vector Machine* (SVM) adalah metode klasifikasi yang menggunakan hyperplane untuk memisahkan data ke dalam kelas yang berbeda. Algoritma ini efektif untuk data berdimensi tinggi dan mampu menangani hubungan non-linear melalui kernel trick. Namun, SVM membutuhkan tuning parameter yang baik untuk performa optimal dan kurang efisien pada dataset besar[3].  Berikut mekanisme kerja algoritma SVM:

- SVM melakukan klasifikasi dengan cara menentukan batas keputusan yang memisahkan data antar kelas yang memiliki jarak terdekat dengan semua data pada kelas. Batas keputusan  yang dibuat oleh SVM disebut dengan *maximum margin classifier* dan dipisahkan oleh suatu bidang hiper (*hyperplane*) [4].
- SVM mencari pemisah terbaik (*hyperplane*) antara dua kelas dengan memetakan data ke ruang dimensi tinggi. Jika data tidak dapat dipisahkan secara linier, maka SVM akan melakukan transformasi non-linier ke dalam dimensi yang lebih tinggi menggunakan *kernel trick*


**Parameter**

Model SVM dalam proyek ini dikonfigurasi menggunakan parameter sebagai berikut:

* `C=10`: Parameter regularisasi yang mengontrol trade-off antara margin maksimal dan kesalahan klasifikasi. Nilai C yang lebih besar (seperti 10) memberikan penalti yang lebih tinggi terhadap kesalahan klasifikasi, sehingga model akan mencoba mengklasifikasikan semua data dengan benar namun bisa berisiko overfitting.
* `kernel='rbf'`: Jenis kernel yang digunakan adalah *Radial Basis Function* (RBF), yang umum digunakan untuk data yang tidak dapat dipisahkan secara linier. Kernel ini memetakan data ke dimensi yang lebih tinggi untuk menemukan *hyperplane* yang optimal.
* `gamma='scale'`: Parameter gamma mengontrol pengaruh satu titik data terhadap model. Dengan opsi `'scale'`, nilai gamma ditentukan secara otomatis berdasarkan `1 / (n_features * X.var())`, yang cenderung memberikan hasil yang stabil untuk banyak kasus.
* `random_state=32`: Seed acak untuk memastikan bahwa hasil pelatihan dapat direproduksi.


### Random Forest Classification

*Random Forest Classification* adalah algoritma ensemble learning berbasis decision tree yang menggabungkan banyak pohon keputusan (*decision tree*) untuk menghasilkan prediksi yang lebih akurat dan stabil. Model ini memiliki keunggulan dalam mengatasi overfitting yang umum terjadi pada decision tree tunggal. Berikut cara kerja algoritma *Random Forest Classification*:

- Membuat sampel bootstrap dari dataset asli
- Membangun atau melatih beberapa pohon keputusanuntuk setiap sampel, menggunakan subset fitur acak pada setiap split
- Melakukan voting dari hasil prediksi setiap pohon dan mengambil kelas mayortias dari hasil voting setiap pohon.

**Parameter**

Model *Random Forest* dalam proyek ini dikonfigurasi menggunakan parameter sebagai berikut:

* `bootstrap=True`: Menentukan bahwa sampel data untuk setiap pohon akan diambil menggunakan teknik *bootstrap sampling*, yaitu pengambilan sampel acak dengan pengembalian dari dataset asli.
* `criterion='gini'`: Menggunakan indeks Gini sebagai fungsi pengukuran untuk memisahkan node dalam pohon keputusan. Gini menghitung ketidakteraturan (impurity) suatu node; semakin rendah nilai Gini, semakin bersih pemisahannya.
* `max_features='sqrt'`: Menentukan jumlah fitur maksimum yang dipertimbangkan untuk split di setiap node. Dengan nilai `'sqrt'`, hanya akar kuadrat dari jumlah total fitur yang dipertimbangkan, yang membantu mengurangi korelasi antar pohon dan memperkuat ensemble.
* `min_samples_leaf=1`: Menentukan jumlah minimum sampel yang diperlukan untuk berada di daun (leaf node). Nilai 1 berarti node daun bisa dibentuk hanya dengan satu data.
* `min_samples_split=2`: Jumlah minimum sampel yang diperlukan untuk memisahkan node internal. Nilai 2 berarti node akan dipisah jika memiliki dua atau lebih data.
* `n_estimators=100`: Jumlah pohon keputusan yang digunakan dalam ensemble. Semakin banyak jumlah pohon, biasanya semakin stabil dan akurat prediksinya, namun juga meningkatkan waktu komputasi.
* `random_state=32`: Seed pengacakan untuk memastikan hasil pelatihan dapat direproduksi.


### XGBoost

XGBoost adalah salah satu algoritma boosting yang sangat powerful dan sering digunakan dalam kompetisi data science karena kemampuannya dalam menghasilkan model yang sangat akurat. Keunggulan XGBoost meliputi efisiensi waktu pelatihan, pengendalian overfitting, dan fleksibilitas terhadap berbagai jenis data.

**Parameter**

Model XGBoost dalam proyek ini dikonfigurasi dengan parameter sebagai berikut:

* `use_label_encoder=False`: Menonaktifkan encoder label bawaan XGBoost untuk mencegah warning terkait versi terbaru. Label harus sudah dalam bentuk numerik.
* `eval_metric='logloss'`: Metode evaluasi yang digunakan saat pelatihan adalah log loss (*logarithmic loss*), yang cocok untuk klasifikasi biner atau multikelas.
* `learning_rate=0.2`: Menentukan seberapa besar langkah yang diambil model pada setiap iterasi boosting. Nilai ini lebih besar dari default (0.1), artinya model belajar lebih cepat, tetapi berpotensi overfitting jika tidak dikombinasikan dengan pengaturan lain yang tepat.
* `max_depth=10`: Menentukan kedalaman maksimum dari pohon keputusan. Nilai yang lebih besar memungkinkan model menangkap lebih banyak kompleksitas dalam data, tetapi juga meningkatkan risiko overfitting.
* `n_estimators=100`: Jumlah pohon yang akan dibangun oleh model. Meningkatkan jumlah estimator bisa meningkatkan akurasi, tetapi juga meningkatkan waktu pelatihan dan risiko overfitting.
* `subsample=0.8`: Proporsi sampel data pelatihan yang digunakan untuk membangun setiap pohon. Nilai di bawah 1.0 membantu mencegah overfitting dengan memperkenalkan variasi antar pohon.
* `random_state=32`: Menentukan seed untuk pengacakan, agar hasil pelatihan bisa direproduksi.


### Hyperparameter Tuning

Hyperparameter tuning adalah proses penting dalam pengembangan model *machine learning* untuk meningkatkan performa prediksi. Hyperparameter merupakan parameter eksternal yang mengontrol proses pelatihan model dan tidak dipelajari langsung dari data. Contohnya meliputi jumlah pohon dalam *Random Forest*, nilai C dalam SVM, atau *learning rate* dalam XGBoost. 

Tujuan dari hyperparameter tuning adalah untuk menemukan kombinasi parameter terbaik yang menghasilkan model dengan performa optimal pada data validasi, tanpa menyebabkan *overfitting* atau *underfitting*. Proses ini bertujuan untuk memaksimalkan metrik evaluasi seperti *accuracy*, *precision*, *recall*, atau *f1-score*.

Dalam proyek ini, proses tuning dilakukan menggunakan metode *Grid Search*. *Grid Search* adalah pendekatan *exhaustive search* yang mencoba semua kombinasi nilai hyperparameter yang telah ditentukan dalam sebuah grid. Untuk setiap kombinasi, model dilatih dan dievaluasi menggunakan teknik validasi silang (*cross-validation*) untuk memastikan hasil tuning lebih general terhadap data yang belum terlihat.

Contoh konfigurasi grid search yang digunakan dalam proyek ini antara lain:

* **SVM**

  * `C`: \[0.1, 1, 10]
  * `kernel`: \['linear', 'rbf']
  * `gamma`: \['scale', 'auto']

* **Random Forest Classifier**

  * `n_estimators`: \[100, 200, 300]
  * `max_depth`: \[None, 10, 20, 30]
  * `min_samples_split`: \[2, 10]
  * `min_samples_leaf`: \[1, 5]

* **XGBoost**

  * `n_estimators`: \[100, 200]
  * `learning_rate`: \[0.01, 0.1, 0.2]
  * `max_depth`: \[3, 6, 9]
  * `subsample` : \[0.8, 1]



Berikut hasil dari proses Hyperparameter Tuning menggunakan metode *Grid Search*:

| Model                  | Best Params                                                     | Best Accuracy |
|------------------------|-----------------------------------------------------------------|---------------|
| SupportMachineClassifier | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf', random_state=32}                                           | 0.960348     |
| RandomForestClassifier   | {'bootstrap': True, 'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, random_state=32}| 0.991296      |
| XGBClassifier          | {'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 100, 'subsample': 0.8, random_state=32}                     | 0.993230      |



## Evaluation

Tahapan evaluasi bertujuan untuk mengukur kinerja model dalam melakukan prediksi terhadap data yang belum pernah dilihat sebelumnya (data uji). Evaluasi dilakukan setelah model dilatih dan disetel hyperparameternya melalui proses hyperparameter tuning. Pada proyek ini, model dievaluasi berdasarkan kemampuannya dalam melakukan klasifikasi biner, yaitu memprediksi apakah suatu permohonan pinjaman akan disetujui atau ditolak.

### **Metrik Evaluasi**

Beberapa metrik evaluasi yang digunakan dalam proyek ini antara lain:



 **1. Accuracy (Akurasi)**

Mengukur proporsi prediksi yang benar terhadap seluruh data uji. Meskipun populer, akurasi bisa menyesatkan jika data tidak seimbang (misalnya, lebih banyak data pinjaman yang ditolak dibanding disetujui).

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

* **TP**: True Positive
* **TN**: True Negative
* **FP**: False Positive
* **FN**: False Negative



 **2. Precision (Presisi)**

Mengukur seberapa banyak prediksi positif yang benar-benar positif. Presisi penting ketika kesalahan dalam menyetujui pinjaman yang seharusnya ditolak harus diminimalkan.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$



 **3. Recall (Sensitivitas / True Positive Rate)**

Mengukur seberapa banyak data positif yang berhasil dikenali dengan benar. Recall penting jika tujuan adalah mengidentifikasi semua permohonan pinjaman yang layak.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$



 **4. F1-Score**

Merupakan rata-rata harmonik dari precision dan recall. Metrik ini berguna saat dibutuhkan keseimbangan antara *false positive* dan *false negative*.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$



 **5. Confusion Matrix**

Matriks ini menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas (misalnya: disetujui dan ditolak), memberikan wawasan lebih mendalam terhadap jenis kesalahan yang dilakukan oleh model. Format umum:

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | ------------------- | ------------------- |
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

 **6. ROC AUC (Receiver Operating Characteristic – Area Under the Curve)**

ROC AUC digunakan untuk mengukur performa model klasifikasi pada berbagai ambang batas (threshold). ROC adalah grafik yang menampilkan hubungan antara *True Positive Rate* (Recall) dan *False Positive Rate* (FPR). Sedangkan **AUC** (Area Under the Curve) mengukur luas area di bawah kurva ROC — semakin besar nilai AUC (maksimum 1), semakin baik performa model.

* **True Positive Rate (TPR)** = Recall

$$
\text{TPR} = \frac{TP}{TP + FN}
$$

* **False Positive Rate (FPR)**

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

Nilai **AUC** berkisar antara 0 hingga 1:

* AUC = 1: model sempurna
* AUC > 0.9: sangat baik
* AUC = 0.5: model tidak lebih baik dari tebakan acak
* AUC < 0.5: model salah prediksi (kebalikannya benar)

### Hasil Evaluasi

#### *Summary Table*

Tabel berikut menyajikan ringkasan performa dari tiga model klasifikasi: XGBoost *Classifier*, *Random Forest Classifier*, dan *Support Vector Classifier* berdasarkan metrik evaluasi utama:

|  Model                    | *Accuracy*  | *Precision* | *Recall* | *F-Score* | ROC AUC  |
| ------------------------- | ----------- | ----------- | ---------| --------- | -------- |
| XGB Classifier            | 0.993230    | 1.000000    | 0.986460 | 0.993184  | 0.999892 |
| Random Forest Clasifier   | 0.991296	  | 1.000000    | 0.982592 | 0.991220  | 0.999981 |
| Support Vector Classifier | 0.960348    | 0.977912    | 0.941973 | 0.959606  | 0.991994 |

Dari tabel di atas, dapat disimpulkan sebagai berikut.
- XGB *Classifier* memberikan performa terbaik dengan *accuracy* dan ROC AUC tertinggi.
- *Random Forest* juga menunjukkan performa sangat baik dan hampir setara dengan XGBoost.
- *Support Vector Classifier* (SVC) berada di posisi ketiga, dengan nilai metrik sedikit lebih rendah, namun tetap sangat kompetitif.

#### *Model Comparasion*

<p align="center">
<img src="https://github.com/user-attachments/assets/a2921513-d364-4136-ae6e-19f5779fd47f" alt="comparasion" />
</p><div align="center">Gambar 11 - Perbandingan Performa Setiap Model</div>

Visualisasi di atas memperjelas bagaimana setiap model memiliki performa yang konsisten tinggi. XGBoost dan *Random Forest* unggul pada semua metrik, sedangkan SVC sedikit tertinggal dalam *recall* dan *F1-score*. 

#### *Classification Report*

Classification report menyajikan hasil metrik per kelas (0 = Ditolak, 1 = Disetujui), memungkinkan kita memahami seberapa baik model mengenali masing-masing label. Berikut adalah *classification report* dari setiap model.

* ***Support Vector Classifier***
  
  ```python
    Classification Report:
                precision    recall  f1-score   support
  
             0       0.94      0.98      0.96       517
             1       0.98      0.94      0.96       517
  
      accuracy                           0.96       1034
     macro avg       0.96      0.96      0.96       1034
  weighted avg       0.96      0.96      0.96       1034
  ```

* ***Random Forest Classifier***
  ```python
    Classification Report:
                precision    recall  f1-score   support
  
             0       0.98      1.00      0.99       517
             1       1.00      0.98      0.99       517
  
      accuracy                           0.99       1034
     macro avg       0.99      0.99      0.99       1034
  weighted avg       0.99      0.99      0.99       1034
  ```
* ***XGB Classifier***

  ```python
    Classification Report:
                precision    recall  f1-score   support
  
             0       0.99      1.00      0.99       517
             1       1.00      0.99      0.99       517
  
      accuracy                           0.99       1034
     macro avg       0.99      0.99      0.99       1034
  weighted avg       0.99      0.99      0.99       1034
  ```

#### *Confusion Matrix*

Confusion Matrix adalah tabel yang menggambarkan kinerja model klasifikasi dengan membandingkan antara hasil prediksi model dengan label sebenarnya dari data uji. Matriks ini berisi empat komponen utama yaitu *True Positive* (TP), *False Positive* (FP), *False Negative* (FN), *True Negative* (TN). Berikut *confuison matrix* dari setiap model.

* ***Support Vector Classifier***
  
  <p align="center">
  <img src="https://github.com/user-attachments/assets/cf8cf695-77cd-4837-b0d0-bf6789aa124f" alt="cm_svm" />
  </p><div align="center">Gambar 12 - Confusion Matrix SVM</div>
  
* ***Random Forest Classifier***
  
  <p align="center">
  <img src="https://github.com/user-attachments/assets/f58955eb-b4fe-4749-b527-fbd63b81411a" alt="cm_rf" />
  </p><div align="center">Gambar 13 - Confusion Matrix Random Forest</div>
  
* ***XGB Classifier***

  <p align="center">
  <img src="https://github.com/user-attachments/assets/bd58b946-165f-40dc-b296-329997576d18" alt="cm_xgb" />
  </p><div align="center">Gambar 14 - Confusion Matrix XGBoost</div>

#### Kurva ROC AUC 

<p align="center">
<img src="https://github.com/user-attachments/assets/b2986072-b081-4b65-a707-7c9599d37eb2" alt="roc_auc" />
</p><div align="center">Gambar 15 - ROC AUC Plot</div>

Berikut informasi atau *insight* yang didapat dari kurva ROC AUC di atas.
- Kurva ROC mendekati sudut kiri atas, menunjukkan True Positive Rate tinggi dan False Positive Rate rendah.
- Nilai AUC > 0.99 untuk ketiga model, menandakan kemampuan diskriminatif yang sangat baik (mampu membedakan antara kelas "disetujui" dan "ditolak").

#### *Feature Importance*

Selain nilai evaluasi di atas, pada proyek ini juga dilakukan proses analisis fitur paling berpengaruh dalam melakukan prediksi. Hal ini dilakukan untuk melihat faktor apa saja yang dapat mempengaruhi proses prediksi kelayakan calon nasabah. Berikut hasil analisis *feature importance* dari dua model terbaik yang dibangun pada proyek ini.

* *Random Forest Classifier*
  
  | No. | Feature                    | Importance |
  | --- | -------------------------- | ---------- |
  | 1   | cibil\_score               | 0.829835   |
  | 2   | loan\_term                 | 0.049259   |
  | 3   | loan\_amount               | 0.030204   |
  | 4   | income\_annum              | 0.017469   |
  | 5   | luxury\_assets\_value      | 0.017071   |
  | 6   | bank\_asset\_value         | 0.014885   |
  | 7   | residential\_assets\_value | 0.014806   |
  | 8   | commercial\_assets\_value  | 0.014462   |
  | 9   | no\_of\_dependents         | 0.007608   |
  | 10  | self\_employed             | 0.002567   |
  | 11  | education                  | 0.001834   |


* XGBoost *Classifier*
  
  | No. | Feature                    | Importance |
  | --- | -------------------------- | ---------- |
  | 1   | cibil\_score               | 47.943947  |
  | 2   | loan\_term                 | 10.541603  |
  | 3   | loan\_amount               | 2.017761   |
  | 4   | income\_annum              | 1.802001   |
  | 5   | luxury\_assets\_value      | 1.081965   |
  | 6   | bank\_asset\_value         | 0.792145   |
  | 7   | self\_employed             | 0.770378   |
  | 8   | residential\_assets\_value | 0.743889   |
  | 9   | no\_of\_dependents         | 0.679367   |
  | 10  | commercial\_assets\_value  | 0.643748   |
  | 11  | education                  | 0.401160   |

Terlihat pada kedua tabel bahwa fitur paling berpengaruh dalam proses prediksi adalah fitur `cibil_score` yang memiliki nilai *importance* paling tinggi dan juga jauh lebih besar dibanding fitur lainnya. Selanjutnya ada fitur `loan_term`, `loan_ammount` sebagai fitur dengan nilai *importance* di urutan kedua dan ketiga.

## Kesimpulan

Berdasarkan hasil eksperimen dan evaluasi terhadap ketiga model machine learning (Support Vector Classifier, Random Forest Classifier, dan XGBoost Classifier), proyek ini berhasil menjawab pertanyaan-pertanyaan utama dalam *problem statement* sebagai berikut:

* Prediksi kelayakan pinjaman dapat dilakukan secara efektif dengan memanfaatkan model *machine learning* yang dilatih menggunakan data historis nasabah. Dalam proyek ini, data berupa informasi demografis dan keuangan calon nasabah digunakan sebagai fitur input untuk membangun model klasifikasi biner yang memutuskan apakah pinjaman akan *disetujui* (kelas 1) atau *ditolak* (kelas 0). Proses pelatihan model dilakukan dengan pendekatan supervised learning, menggunakan data yang telah dilabeli berdasarkan keputusan sebelumnya.

* Berdasarkan analisis *feature importance* dari dua model terbaik, yaitu **XGBoost Classifier** dan **Random Forest Classifier**, diketahui bahwa faktor utama yang paling memengaruhi keputusan persetujuan pinjaman adalah:
  * `cibil_score`
  Merupakan fitur paling berpengaruh dalam kedua model, menandakan bahwa skor kredit calon nasabah sangat menentukan kelayakan pinjaman.

  * `loan_term` dan `loan_amount`
    Dua fitur ini juga memiliki kontribusi signifikan, menunjukkan bahwa jangka waktu dan jumlah pinjaman menjadi pertimbangan penting dalam menentukan risiko.

  * `income_annum` dan `luxury_assets_value`
    Meskipun tidak sekuat tiga fitur utama sebelumnya, informasi penghasilan tahunan dan aset mewah turut memberikan pengaruh terhadap keputusan akhir.

  Sementara itu, fitur seperti `education`, `self_employed`, dan `no_of_dependents` menunjukkan pengaruh yang lebih rendah, sehingga dapat dikatakan memiliki kontribusi minor terhadap hasil prediksi kelayakan.
* Berdasarkan hasil evaluasi, model **XGBoost Classifier** memberikan performa terbaik di antara ketiga model yang diuji, dengan nilai *accuracy* sebesar **0.993**, *F1-score* sebesar **0.993**, dan skor ROC AUC mendekati sempurna (**0.9999**). Ini menunjukkan bahwa XGBoost sangat efektif dalam mengklasifikasikan pengajuan pinjaman secara akurat, baik untuk kelas yang disetujui maupun yang ditolak. *Random Forest Classifier* juga menunjukkan performa yang sangat kompetitif, hanya sedikit di bawah XGBoost. Sementara itu, *Support Vector Classifier* meskipun memiliki akurasi dan F1-score yang baik, masih tertinggal dibandingkan dua model lainnya dalam hal *recall*.




## Referensi

[1] S. P. Dewi, “Pengaruh *Capital Adequacy Ratio*, *Non Performing Loan*, *Loan to Deposit Ratio* dan Efisiensi Operasional Terhadap Profitabilitas Perbankan Yang Terdaftar Di Bursa Efek Indonesia,” J. Akunt., vol. 18, no. 3, pp. 422–437, 2014.

[2] D. S. Bhakti, A. Prasetyo, and P. Arsi, "*Implementation of Hyperparameter Tuning in Random Forest Algorithm for Loan Approval Prediction*," Jurnal Teknik Informatika (JUTIF)., vol. 5, no. 4, pp. 63-69. 2024

[3] A. Kurani, P. Doshi, A. Vakharia, and M. Shah, “*A Comprehensive Comparative Study of Artificial Neural Network (ANN) and Support Vector Machines (SVM) on Stock Forecasting*,” Ann. Data Sci., vol. 10, no. 1, pp. 183–208, 2023, doi: https://doi.org/10.1007/s40745-021-00344-x.

[4] K. Kristiawan and A. Widjaja, “Perbandingan Algoritma *Machine Learning* dalam Menilai Sebuah Lokasi Toko Ritel,” Jurnal Teknik Informatika dan Sistem Informasi, vol. 7, no. 1, Art. no. 1, Apr. 2021, doi: https://doi.org/10.28932/jutisi.v7i1.3182.

[5] B. Pernama and H. D. Purnomo, "Analisis Risiko Pinjaman dengan Metode *Support Vector Machine*, *Artificial Neural Network* dan *Naive Bayes*", Jurnal JTIK (Jurnal Teknologi Informasi dan Komunikasi), vol. 7, no. 1, pp. 92 - 99, 2023, doi: https://doi.org/10.35870/jtik.v7i1.693 .

[6] W. Andriani, G. Gunawan, and N. N. P. W. Naja, "Analisis Perbandingan *Machine Learning* untuk Prediksi Kelayakan Kredit Perbankan pada Bank BRI Tegal", Jurnal Penerapan Teknologi, Informasi dan Komunikasi, vol. 4, no. 1, pp. 82-92, 2025, doi: https://doi.org/10.24246/itexplore.v4i1.2025.pp82-92 

[7] N. Hadianto, H. B. Novitasari, and A. Rahmawati, "Klasifikasi Peminjaman Nasabah Bank Menggunakan Metode *Neural Network*", Jurnal PILAR Nusa Mandiri, vol. 15, no. 2, pp. 163-170, 2019, doi: https://doi.org/10.33480/pilar.v15i2.658
