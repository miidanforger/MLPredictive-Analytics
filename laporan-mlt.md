# Laporan Proyek Machine Learning - Muhamad Dani

## Project Overview
Sejak awal 1980-an, konsep manajemen hubungan
(CRM) di bidang pemasaran telah menjadi semakin penting. Memperoleh dan
mempertahankan pelanggan yang paling menguntungkan adalah perhatian serius dari a
perusahaan untuk melakukan kampanye pemasaran yang lebih bertarget (Bult &
Wansbeek, 1995; Hughes, 1994; Hwang, Jung, & Suh, 2004; Kahan,
1998; Malthouse & Blattberg, 2004; Schmittlein, Morrison, &
Kolombo, 1987; Batu, 1995). Untuk hubungan pelanggan yang efektif
manajemen, penting untuk mengumpulkan informasi tentang pelanggan
nilai. Model paling kuat dan paling sederhana untuk mengimplementasikan CRM
mungkin model RFM – Keterkinian, Frekuensi, dan nilai Moneter.
RFM adalah model berbasis perilaku, artinya digunakan untuk menganalisis
perilaku yang dilakukan pelanggan, dan membuat prediksi berdasarkan
pada perilaku ini (Colombo & Jiang, 1999; Hughes, 1996).
RFM memiliki akibat wajar: Pelanggan yang telah membeli atau berkunjung
baru-baru ini, lebih sering, atau menciptakan nilai moneter yang lebih tinggi
lebih mungkin merespons upaya pemasaran Anda, dibandingkan dengan pelanggan lain yang lebih jarang, lebih jarang,
dan menciptakan lebih sedikit nilai moneter.

## Business Understanding

### Problem Statements

- Bagaimana cara membangun sistem klasifikasi untuk melakukan diagnosa Darah terbaik?

### Goals

- Dapat mengetahui cara membangun sistem klasifikasi untuk melakukan Transfusi dengan model terbaik.

### Solution statements.

- Menawarkan solusi sistem diagnosa dengan metode klasifikasi. Untuk mendapatkan solusi terbaik akan digunakan tiga model yang berbeda (KNN, Random Forest, AdaBoosting) dengan *Hyperparameter tuning*. Selain itu, untuk mengukur kinerja model digunakan metrik akurasi. Di mana model terbaik nantinya harus memperoleh nilai akurasi tertinggi dari data uji.

## Data Understanding

Tabel 1. Informasi Dataset

| | Keterangan |
|---|---|
| Sumber | [Kaggle - Blood Transfusion Dataset](https://www.kaggle.com/datasets/whenamancodes/blood-transfusion-dataset) |
| Jumlah Data | 748 |
| *Usability* | 10.0 |
| *Rating* | *gold* |
| Jenis dan Ukuran Berkas | csv (3 kB) |

### Variabel-variabel pada Dataset

Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/whenamancodes/blood-transfusion-dataset), variabel-variabel pada Diabetes dataset adalah sebagai berikut:

-Recency: bulan sejak transfusi darah terakhir
-Frequency: banyak transfusi darah
-Monetary: total banyak darah yang didonorkan
-Time: bulan sejak transfusi darah pertama
-Class: apakah pendonor darah (1) pada maret 2007 atau tidak (0)

### Menangani Missing Value

Untuk mendeteksi *missing value* digunakan fungsi isnull().sum() dan diperoleh:

Tabel 2. Hasil Deteksi *Missing Value*

| Kolom | Jumlah *Missing Value* |
|---|:---:|
| Recency  | 0 |
| Glucose | 0 |
| Frequency  | 0 |
| Monetary  | 0 |
| Time  | 0 |
| Class | 0 |
Dari Tabel 2 di atas, terlihat bahwa setiap fitur tidak memiliki *missing value*.

### Univariate Analysis

Selanjutnya, untuk fitur numerik, akan dilakukan visualisasi dengan histogram pada masing-masing fiturnya sebagai berikut.

![](https://miidan.github.io/images/histogram.png)

Gambar 1. Histogram pada Setiap Fitur Numerik

Berdasarkan Gambar 1. di atas, diperoleh beberapa informasi, antara lain:
- Pada histogram Frequency dan histogram Monetary miring ke kiri (left-skewed). Hal ini akan berimplikasi pada model.

### Multivariate Analysis

Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi pairplot(), dengan output sebagai berikut.

![](https://miidan.github.io/images/pairplot.png)

Gambar 2. Visualisasi Hubungan antar Fitur Numerik

Pada pola sebaran data grafik pairplot di atas, terlihat fitur Frequency memiliki korelasi cukup kuat (positif) dengan fitur Pregnancies. Untuk mengevaluasi skor korelasinya, akan digunakan fungsi corr() sebagai berikut.

![](https://miidan.github.io/images/correlation.png)

Gambar 3. Korelasi antar Fitur Numerik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

## Data Preparation

### Reduksi Dimensi dengan PCA

PCA umumnya digunakan ketika variabel dalam data yang memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. Sebelumnya perlu cek kembali korelasi antar fitur (selain fitur target) dengan menggunakan pairplot.

![](https://miidan.github.io/images/pairplot-pca.png)

Gambar 4. Visualisasi Hubungan antar Fitur Selain Fitur Target (Outcome)

Selanjutnya kita akan mereduksi fitur Time, Frequency dan fitur Monetary karena ketiganya berkorelasi cukup kuat yang dapat dilihat pada visualisasi pairplot di atas.

Untuk implementasinya menggunakan fungsi PCA() dari sklearn dengan mengatur nilai parameter n_components sebanyak fitur yang akan dikenakan PCA.

Tabel 3. Proporsi *Principal Component* dari Hasil PCA

| PC Pertama | PC Kedua |
|:---:|:---:|
| 0.848 | 1 |

Arti dari output di atas adalah, 100% informasi pada ketiga fitur (Time, Frequency dan Monetary) terdapat pada PC (Principal Component) pertama. Sedangkan sisanya sebesar 0% terdapat pada PC kedua dan ketiga

Berdasarkan hasil tersebut, kita akan mereduksi fitur dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur yang menggantikan tiga fitur lainnya (Time, Frequency dan Monetary). Kita beri nama fitur ini PCA_TFM.

Tabel 4. Tampilan 5 Sampel dari Dataset Setelah Dilakukan Reduksi Fitur

|index|Recency \(months\)|PCA\_TFM|
|---|---|---|
|239|-0\.23254060642268404|0\.5639927112736453|
|269|0\.36325751971598386|-0\.27877415230253577|
|151|-0\.7091791073336183|0\.05835546402298325|
|245|-0\.7091791073336183|-0\.11009498966454298|
|59|-0\.8283387325613519|0\.05822681523834582|

### Train Test Split

Pada tahap ini akan dibagi dataset menjadi data latih (train) dan data uji (test). Pada kasus ini akan menggunakan proporsi pembagian sebesar 80:20 dengan fungsi train_test_split dari sklearn.

Tabel 5. Jumlah Data Latih dan Uji

| Jumlah Data Latih | Jumlah Data Uji | Jumlah Total Data |
|:---:|:---:|:---:|
| 598 | 150 | 748 |

### Standarisasi

Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan mean sama dengan 0.

Tabel 6. Hasil Proses Standarisasi pada Setiap Fitur

|index|Recency \(months\)|PCA\_TFM|
|---|---|---|
|239|-0\.23254060642268404|0\.5639927112736453|
|269|0\.36325751971598386|-0\.27877415230253577|
|151|-0\.7091791073336183|0\.05835546402298325|
|245|-0\.7091791073336183|-0\.11009498966454298|
|59|-0\.8283387325613519|0\.05822681523834582|

## Modeling
Pada tahap ini, kita akan menggunakan tiga algoritma untuk kasus klasifikasi ini. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:

1. K-Nearest Neighbor

    Kelebihan algoritma KNN adalah mudah dipahami dan digunakan sedangkan kekurangannya kika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias.

2. Random Forest
    
    Kelebihan algoritma Random Forest adalah menggunakan teknik Bagging yang berusaha melawan *overfitting* dengan berjalan secara paralel. Sedangkan kekurangannya ada pada kompleksitas algoritma Random Forest yang membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma seperti Decision Tree.

3. Boosting Algorithm

    Kelebihan algoritma Boosting adalah menggunakan teknik Boosting yang berusaha menurunkan bias dengan berjalan secara sekuensial (memperbaiki model di tiap tahapnya). Sedangkan kekurangannya hampir sama dengan algoritma Random Forest dari segi kompleksitas komputasi yang menjadikan waktu pelatihan relatif lebih lama, selain itu noisy dan outliers sangat berpengaruh dalam algoritma ini.

Untuk langkah pertama, kita akan siapkan DataFrame baru untuk menampung nilai metrik Akurasi pada setiap model / algoritma. Hal ini berguna untuk melakukan analisa perbandingan antar model.

### Model KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih k tetangga terdekat. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika kita memilih k yang terlalu rendah, maka akan menghasilkan model yang *overfitting* dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k yang terlalu tinggi, maka model yang dihasilkan akan *underfitting* dan prediksinya memiliki bias yang tinggi [[3]](https://learning.oreilly.com/library/view/machine-learning-with/9781617296574/).

Oleh karena itu, kita akan mencoba beberapa nilai k yang berbeda (1 sampai 20) kemudian membandingan mana yang menghasilkan nilai metrik model (pada kasus ini kita pakai akurasi) terbaik. Selain itu, kita akan menggunakan metrik ukuran jarak secara default (Minkowski Distance) pada *library* sklearn.

Tabel 7. Perbandingan Nilai K terhadap Akurasi

| K | Akurasi |
|:---:|---|
| 1 | 0.64 |
| 2 | 0.6666666666666666 |
| 3 | 0.6533333333333333 |
| 4 | 0.7333333333333333 |
| 5 | 0.7 |
| 6 | 0.76 |
| 7 | 0.7333333333333333 |
| 8 | 0.74 |
| 9 | 0.7466666666666667 |
| 10 | 0.7333333333333333 |
| 11 | 0.7333333333333333 |
| 12 | 0.7266666666666667 |
| 13 | 0.7266666666666667 |
| 14 | 0.7333333333333333 |
| 15 | 0.7266666666666667 |
| 16 | 0.72 |
| 17 | 0.72 |
| 18 | 0.7133333333333334 |
| 19 | 0.7333333333333333 |
| 20 | 0.7333333333333333 |

Jika divisualisasikan dengan fungsi `plot()` diperoleh:

![](https://miidan.github.io/images/tunning-k-accuracy.png)

Gambar 5. Visualisai Nilai K terhadap Akurasi

Dari hasil output diatas, nilai akurasi terbaik dicapai ketika k = 6 yaitu sebesar 0.76. Oleh karena itu kita akan menggunakan k = 6 dan menyimpan nilai akurasi nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam df_models yang telah kita siapkan sebelumnya.

### Model Random Forest

Random forest merupakan algoritma *supervised learning* yang termasuk ke dalam kategori *ensemble* (group) learning. Pada model *ensemble*, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model *ensemble* ini digabungkan untuk membuat prediksi akhir. Jenis metode *ensemble* yang digunakan pada Random Forest adalah teknik Bagging. Metode ini bekerja dengan membuat subset dari data train yang independen. Beberapa model awal (base model / weak model) dibuat untuk dijalankan secara simultan / paralel dan independen satu sama lain dengan subset data train yang independen. Hasil prediksi setiap model kemudian dikombinasikan untuk menentukan hasil prediksi final. 

Parameter-parameter (*hyperparameter*) yang digunakan pada algoritma ini antara lain:

- n_estimator: jumlah trees (pohon) di forest.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.

Untuk menentukan nilai *hyperparameter* (n_estimator & max_depth), dilakukan tuning dengan GridSearchCV dan hasilnya sebagai berikut:

Tabel 8. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan Random Forest

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 90 |
| max_depth | 4, 8, 16, 32 | 32 |
| Accuracy data latih | | 0.7976588628762542 |
| Accuracy data uji | | 0.7466666666666667 |

Dari hasil output di atas diperoleh nilai Akurasi terbaik dalam jangkauan parameter params_rf yaitu 0.7976 (dengan data train) dan 0.7466 (dengan data test) dengan n_estimators: 90 dan max_depth: 32. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai Akurasi nya kedalam df_models yang telah kita siapkan sebelumnya.

### Model AdaBoosting

Jika sebelumnya kita menggunakan algoritma *bagging* (Random Forest). Selanjutnya kita akan menggunakan metode lain dalam model *ensemble* yaitu teknik *Boosting*. Algoritma *Boosting* bekerja dengan membangun model dari data train. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Teknik ini bekerja secara sekuensial.

Pada kasus ini kita akan menggunakan metode *Adaptive Boosting*. Untuk implementasinya kita menggunakan AdaBoostClassifier dari library sklearn dengan base_estimator defaultnya yaitu DecisionTreeClassifier hampir sama dengan RandomForestClassifier bedanya menggunakan metode teknik *Boosting*.

Parameter-parameter (hyperparameter) yang digunakan pada algoritma ini antara lain:

- n_estimator: jumlah *estimator* dan ketika mencapai nilai jumlah tersebut algoritma Boosting akan dihentikan.
- learning_rate: bobot yang diterapkan pada setiap *classifier* di masing-masing iterasi Boosting.
- random_state: digunakan untuk mengontrol *random number* generator yang digunakan.

Untuk menentukan nilai *hyperparameter* (n_estimator & learning_rate) di atas, kita akan melakukan *tuning* dengan GridSearchCV.

Tabel 9. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan AdaBoosting

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 90 |
| learning_rate | 0.001, 0.01, 0.1, 0.2 | 0.2 |
| Accuracy data latih | | 0.7809 |
| Accuracy data uji | | 0.7466 |

Dari hasil output di atas diperoleh nilai Akurasi terbaik dalam jangkauan parameter params_ab yaitu 0.7809 (dengan data train) dan 0.7466 (dengan data test) dengan n_estimators: 90 dan learning_rate: 0.2. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai Akurasi nya kedalam df_models yang telah kita siapkan sebelumnya.

## Evaluation
Dari proses sebelumnya, telah dibangun dan dilatih tiga model yang berbeda (KNN, Random Forest, Boosting). Selanjutnya perlu mengevaluasi model-model tersebut menggunakan data uji dan metrik yang digunakan dalam kasus ini yaitu akurasi. Hasil evaluasi kemudian disimpan ke dalam df_models.

$$\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$$

Dengan:
- $n_{\text{sample}}$ adalah banyaknya data
- $1(\hat{y}_i = y_i)$ bernilai 1 jika $\hat{y}_i$ nilainya sama dengan $y_i$. Dimana $\hat{y}_i$ adalah hasil prediksi sedangkan $y_i$ adalah nilai yang akan diprediksi (nilai yang sebenarnya).

Berdasarkan DataFrame `df_models` diperoleh:

Tabel 10. Nilai Akurasi pada Setiap Model dengan Data Uji

|index|KNN|RandomForest|Boosting|
|---|---|---|---|
|Train Accuracy|0\.8093645484949833|0\.7859531772575251|0\.7809364548494984|
|Test Accuracy|0\.76|0\.74|0\.7466666666666667|

Untuk memudahkan, dilakukan *plot* hasil evaluasi model dengan *bar chart* sebagai berikut:

![](https://miidan.github.io/images/model-acc.png)

Gambar 6. *Bar Chart* Hasil Evaluasi Model dengan Data Latih dan Uji

Dari gambar di atas, terlihat bahwa, model KNN memberikan nilai Akurasi (pada data uji) yang paling tinggi. Sebelum memutuskan model terbaik untuk melakukan prediksi "Outcome" atau hasil diagnosa terhadap penyakit diabetes. Mari kita coba uji prediksi menggunakan beberapa sampel acak (10) pada data uji.

Tabel 11. Hasil Prediksi dari 10 Sampel Acak

|index\_sample|y\_true|prediksi\_KNN|prediksi\_RF|prediksi\_Boosting|
|---|---|---|---|---|
|414|0|0|0|0|
|370|1|0|0|0|
|263|1|0|0|0|
|379|0|0|0|0|
|294|0|0|0|0|
|170|0|0|0|0|
|511|0|0|0|0|
|696|0|0|0|0|
|529|1|1|0|0|
|250|0|0|0|0|

Dari Tabel 11, terlihat bahwa prediksi dengan Random Forest memberikan hasil yang paling mendekati.

## Conclusion
Berdasarkan hasil evaluasi model di atas, dapat disimpulkan bahwa model terbaik untuk melakukan klasifikasi "Outcome" atau diagnosa penyakit diabetes adalah model Random Forest. Hal ini dilihat dari nilai akurasi pada data uji yang menunjukan bahwa model KNN mempunyai akurasi tertinggi sebesar 0.805 disusul dengan RandomForest (0.790).

## Daftar Referensi
[1] Knowledge discovery on RFM model using Bernoulli sequence. [online] Direktorat P2PTM. Available at: <https://www.sciencedirect.com/science/article/abs/pii/S0957417408004508?via%3Dihub> [Accessed April 2009].
[2] Smith JW, Everhart JE, Dickson WC, Knowler WC, Johannes RS. Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. Proc Annu Symp Comput Appl Med Care. 1988 Nov 9:261–5. PMCID: PMC2245318. Tersedia: [tautan](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf).