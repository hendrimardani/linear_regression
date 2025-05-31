## **Laporan Proyek Machine Learning - Hendri Mardani**
***
### **Domain Proyek**
---
Dataset ini tentang kepadatan rumah dalam suatu wilayah dari tahun 1995 - 2020, dengan rincian rata-rata harga rumah di setiap wilayah tersebut. Dataset ini memiliki perbatasan wilayah di negara **London**, dalam kai ini kita akan menggunakan kasus regresi kenapa? karena dalam penjualan ataupun pembelian akan dihadapkan dengan harga, dalam harga penjualan atau pembelian dalam tiap tahun kadang naik kadang turun dengan ini sering kita sebut nilai _kontinu_ atau nilai angka yang berhubungan dengan regresi bukan sebuah klasifikasi, Secara logika harga rumah tiap tahun pasti naik, tetapi ini belum pasti, bisa jadi dalam suatu kota tersebut dikarenakan padatnya jumlah penduduk ataupun karena hal lain contohnya. Terjadinya inflasi disuatu negara. Untuk itu permasalahan ini menggunakan regresi yang akan memprediksi harga tiap tahun.

### **Business Understanding**
---
Dalam kasus dataset ini, kita akan memperkirakan harga dari tahun ke tahun dengan kepadatan jumlah rumah di setiap tahun dan rumah termahal di tahun tersebut.
1. Problem Statement
    * Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh dengan harga?
    * Apakah wilayah mempengaruhi dengan harga kenaikan rata-rata rumah?
2. Goals
    * Membuat prediksi harga setiap tahun dengan menjadikan fitur targetnya adalah _average_price_
    * Membuat sebuah visualisasi kepadatan rumah dari tahun ke tahun.
* Dengan uraian diatas, tampak masalah kita adalah menggunakan nilai _kontinu_, yang hal ini berhubungan dengan model _regression_.
* Untuk pengevaluasian pada masalah ini menggunakan _Mean Squared Error(MSE)_ atau bisa juga menggunakan _Mean Absolute Error(MAE)_
* Pembuatan model kali ini akan menggunakan model _Lasso , Random Forest Regressor dan Linear Regression_ , kemudian kita akan memlih salahsatu model tersebut untuk dijadikan model dengan meminimalisir nilai kesalahan/_error_ yang seminim mungkin sekaligus membandingkannya dengan metode _deep learning_.

### **Data Understanding**
---
Dataset ini tentang harga rumah disetiap daerah tiap bulan dan juga jumlah kepadatan rumah disetiap tahun yang memiliki **13549 baris dan 7 kolom**. Rincian fitur-fitur yang ada pada dataset ini adalah sebagai berikut:
* Nama dataset Housing In London

**Tabel 1. Deskripsi fitur-fitur dataset**

|NO|    Fitur    |                   Deskripsi                      |
|--|:-----------:|:------------------------------------------------:|
|1.|    date     |Tanggal dan tahun harga rumah di wilayah tersebut |
|2.|    area     |Harga rumah disuatu wilayah tersebut              |
|3.|average_price|Harga rata-rata rumah dalam setiap bulan dan tahun|
|4.|    code     |area code disuatu wilayah dalam satuan (code)     |
|5.| houses_sold |Jumlah rumah yang terjual tiap tahun              |
|6.|no_of_crimes |Jumlah kejahatan dalam tiap bulan disetiap wilayah|
|7.|borough_flag |Wilayah disetiap masing-masing rumah (0 dan 1)    |

Link dataset : [dataset](https://www.kaggle.com/datasets/justinas/housing-in-london)

### **Data Preparation**
---

Pada tahapan ini kita akan _mengcleaning_ data, karena sudah dipastikan dalam data tersebut memliki nilai kosong atau yang sering disebut _NaN_, untuk itu kita memeriksa terlebih dahulu apakah ada data yang bernilai _NaN_ atau tidak.

|  #  |     Column    | Non-Null Count | Dtype   |
|:---:|:-------------:|----------------|---------|
| --- |     ------    | -------------- | -----   |
|  0  |      date     | 13549 non-null | object  |
|  1  |      area     | 13549 non-null | object  |
|  2  | average_price | 13549 non-null | int64   |
|  3  |      code     | 13549 non-null | object  |
|  4  |  houses_sold  | 13455 non-null | float64 |
|  5  |  no_of_crimes | 7439 non-null  | float64 |
|  6  |  borough_flag | 13549 non-null | int64   |

**Tabel 2. Fitur NaN**

* Pada **gambar 1** terlihat bahwa fitur **houses_sold, no_of_crimes_ dan borough_flag** memiliki nilai NaN, untuk itu kita menghapus fitur-fitur tersebut karena memiliki banyak nilai yang kosong dan tidak akan masuk pada analisis selanjutnya.Namun sebelumnya kita ubah terlebih dahulu tipedata fitur date dari _object_ ke _datetime64[ns]_, karena ini merupakan berhubungan dengan waktu, atau dengan kata lain _time series_

|  #  |     Column    | Non-Null Count | Dtype          |
|:---:|:-------------:|----------------|----------------|
| --- |     ------    | -------------- | -----          |
|  0  |      date     | 13549 non-null | datetime64[ns] |
|  1  |      area     | 13549 non-null | object         |
|  2  | average_price | 13549 non-null | int64          |
|  3  |      code     | 13549 non-null | object         |
|  4  |  houses_sold  | 13455 non-null | float64        |
|  5  |  no_of_crimes | 7439 non-null  | float64        |
|  6  |  borough_flag | 13549 non-null | int64          |

**Tabel 3. Merubah tipedata fitur _date_**

Langkah selanjutnya setiap data pasti memiliki _outlier_ batas ambang dalam sebuah nilai, yang artinya nilai fitur dibawah atau diatas batas ambang yang wajar atau sering kita sebut kuartil bawah dan kuartil atas dan _outlier ini harus dihapus dari data, dengan ini kita akan memvisualisasikan salahsatu fitur tersebut untuk memastikan sebelum dan sesudah menghapus _outlier_. Caranya adalah dengan menampilkan visualisasi dalam bentuk **boxplot** atau **violinplot**, namun dalam kasus ini menggunakan visualisasi dalam bentuk violinplot.

![violin](https://user-images.githubusercontent.com/49816104/194709041-9e7a2846-ffc7-45f9-beec-f19022dbe03c.jpg)

**Gambar 1. _Outlier_**

* Pada **gambar 1** _outlier_ atau batas ambang atas ini berada pada kisaran 0.8 _average_price_, selebihnya diartikan _outlier_ . untuk melihat perbedaannya lihat pada **gambar 2** sesudah proses penghapusan outlier yang dalam hal ini terdeteksi sebagai _outlier_ sebanyak **240 baris**, data ini akan secara otomatis terhapus dari dataset.

![violin_2](https://user-images.githubusercontent.com/49816104/194709426-44778481-0bb3-4cde-a029-057ed7e551db.jpg)


**Gambar 2.Setelah Penghapusan _Outlier_**

* Terlihat bahwa setelah penghapusan _outlier_ puncaknya berada pada 0.8/800000, sesuai dengan perkiraan tadi ada pada kisaran tersebut, dengan ini dataset menjadi bersih/_clearning_ dan bisa melakukan pemrosesan data.

**Note:**

Untuk tahapan pada kali ini teknik penghapusan _outlier_ bermacam-macam, namun dalam hal ini menggunakan teknik _Z-score_ dengan menggunakan rumus sebagai berikut:

## **Z = (x-$ \alpha$ ) / $\varphi$**

Dengan :

* Z = z-score
* x = nilai fitur
* $\alpha$  = rata-rata fitur(_mean_)
* $\varphi$  = standard deviasi fitur

Selanjutnya analisis terhadap fitur _area_  dalam bentuk visualisasi barplot, untuk melihat jumlah rumah di masing-masing wilayah tersebut.

![bar](https://user-images.githubusercontent.com/49816104/194710037-940f9a0a-d5a8-47da-bbe8-1f656d598f4e.jpg)


**Gambar 3. Jumlah rumah di masing-masing wilayah**

* pada **gambar 3** terlihat bahwa semua wilayah memiliki **300** rumah, berbeda dengan 3 wilayah ini : camden, stimster dan chelsea yang memiliki rumah dibawah **300** dibawah rata-rata wilayah yang sebelumnya. 

Untuk melihat kepadatan jumlah rumah dalam setiap tahun kita bisa menggunakan visualisasi dalam bentuk **kdeplot**, dalam artian kita ingin mengetahui pada tahun berapakah jumlah rumah yang paling padat.

![kde](https://user-images.githubusercontent.com/49816104/194710379-c98c1b8f-f928-4ff8-9c22-5e2085e1c43e.jpg)

**Gambar 4. Bentuk kdeplot**

* Pada **gambar 4** bisa disimpulkan bahwa tingkat kepadatan suatu rumah berada pada tahun **1996** sampai **2002** semakin biru kepadatan jumlah tersebut semakin banyak, dalam visualisasi ini juga terlihat jumlah rumah dari tahun ke tahun semakin meningkat namun harga rumah juga semakin meningkat.

Selanjutnya kita melihat jumlah rumah yang paling banyak diduduki disuatu wilayah dengan bantuan visualisasi **barplot**.


![bar_3](https://user-images.githubusercontent.com/49816104/194710665-12905a10-3201-4134-8798-f3d33bd6d640.jpg)

**Gambar 5. Bentuk barplot**

* Pada **gambar 5** jumlah rumah yang paling banyak diduduki berada pada wilayah **1** dengan rata-rata berada pada kisaran **290000** rumah, lain hal dengan wilayah **0** berada pada kisaran **180000** rumah.

![scatter](https://user-images.githubusercontent.com/49816104/194710930-ad753eb5-9022-4bd4-a165-d4890711228a.jpg)

**Gambar 6. Bentuk scatterplot**

* Pada **gambar 6** sesuai dengan prediksi kita, bahwa kepadatan berada pada tahun **1996** sampai **2002**, namun ketika lebih dari tahun **2002** sebaran data mulai terpisah jauh, apalagi pada wilayah  **0** pada tahun **2008** harga rumah paling mahal, berbeda dengan wilaya satunya **1** pada tahun yang sama harganya berbeda jauh sekali, dengan mengetahui sebaran data atau pola seperti ini kita bisa analisis dataset ini dengan model _regression_, karena merupakah nilai _kontinu_ yaitu berkelanjutan.

![pair](https://user-images.githubusercontent.com/49816104/194711507-b583adf4-fe5b-4a48-8c6a-ac05f26197d7.jpg)

**Gambar 7. Hubungan fitur numerik**

* Pada **Gambar 7** kita visualisasikan fitur numerik antara fitur _borough_flag_ dan fitur _average_price_ kedalam bentuk visualisasi **pairplot** , sekaligus kedalam bentuk **heatmap**. Dalam hal ini tidak ada hubungan atau korelasi antara fitur _borough_flag dan _average_price_, dengan ini kita bisa menghapus fitur _borough_flag_ karena tidak memiiki _insight_ atau analisis terhadap dataset ini.Untuk membacanya ketika fitur memiliki nilai mendekati 0 maka tidak ada hubungan atau korelasi dengan fitur lainnya. Untuk menghapus bisa menggunakan bantuan _method_ drop bawaaan _pandas_.

**Note:**

* Kita juga menghapus fitur _area_ pada dataset, karena dalam hal ini tidak termasuk kedalam dataset _training_ ataupun _validasi_, sehingga kita keseluruhan hanya memiliki **2** fitur yaitu fitur _date_ dan fitur _average_price_ atau yang sering kita sebut harga.Akan tetapi kita tidak menghapus fitur _borough_flag_ karena untuk analisis lebih lanjut visualisasi hasil prediksi.

|  date | borough_flag | average_price |
|:-----:|:------------:|:-------------:|
|   0   |       1      |    91449.0    |
|   1   |       1      |    82203.0    |
|   2   |       1      |    79121.0    |
|   3   |       1      |    77101.0    |
|   4   |       1      |    84409.0    |
|  ...  |      ...     |      ...      |
| 13544 |       0      |    249942.0   |
| 13545 |       0      |    249376.0   |
| 13546 |       0      |    248515.0   |
| 13547 |       0      |    250410.0   |
| 13548 |       0      |    247355.0   |

**Tabel 4. Hasil akhir analisis**

Sampai tahap ini dataset yang kita buat mulai dari penghapusan _outlier_ , fitur-fitur yang tidak diperlukan dalam proses anlisis sudah menjadi data _cleaning_, yang artinya dataset kita sudah bersih dari nilai kosong, fitur yang tidak berkorelasi, dll. Untuk selanjutnya kita membagi dataset ini menjadi dataset _train_ dan __validasi_ tujuannya adalah untuk melatih dataset yang ada pada _train_ dan _validasi_ untuk melakukan test ketika model sudah terlatih atau _pre-trained_ nilai yang diambil adalah secara acak akan tetapi porsinya menggunakan _parameter_ `test_size`, untuk menggunakan teknik ini kita bisa menggunakan bantuan dari modul _sklearn method `train_test_split()`_.

* Jumlah baris dan kolom semuanya = 13309
* Jumlah baris dan kolom _training_ = 10647
* Jumlah baris dan kolom test = 2662

### **Modeling**
---
Langkah selanjutkan kita akan membuat model dengan bantuan modul _sklearn_ selanjutnya membuat sebuah voting model untuk melihat _performance_ model untuk dipilih model yang paling bagus diantara ketiga model yang kita buat. Caranya adalah pertama kita buat  _object_ untuk masing-masing model `Random Forest Regressor()`, `Lasso()` dan `Linear Regression()`, untuk nama voting model yaitu `VotingRegressor()`. Parameter yang dibutuhkan hanya `estimators` dengan didalamnya berisi _array_,  masing-masing model memiliki _key_  _valuenya_ masing-masing.

![voting](https://user-images.githubusercontent.com/49816104/194713265-ad049395-9696-4462-88da-f59fec05b927.png)


**Gambar 8. Hasil _output_ `VotingRegressor()`**

* Cara kerja masing-masing model
    * `Random Forest()` memiliki banyak keputusan dalam membuat prediksi atau dengan katalain gabungan dari model _decision tree_, prediksi ini diambil dari nilai rata-rata keputusan _decision tree_ paling banyak.
    * `Lasso()` hampir sama seperti _linear regression_, hanya saja _lasso_ memiliki parameter alpha, sedangkan _linear regression_ tidak.
    * `LinearRegression` membuat titik di sumbu x atau sering kita sebut _slope_ kemudian membuat garis dengan berbentuk linear mengikuti pola data yang ada

* Setelah membuat _objek_ masing-masing model kita _training dengan memanggil _method `fit()` pada _voting regressor_  dalam hal ini kita evaluasi menggunakan **_loss mean_absolute_error_** akan tetapi terlebih dahulu kita lakukan _standarisasi_ pada data _training_ dengan bantuan modul _sklearn_, tujuannya adalah mengecilkan nilai-nilai yang ada pada fitur pada rentang -1 sampai 1 dan juga memudahkan model dalam menemukan pola

* Untuk rumus **Standarisasi** sama seperti rumus **_z-score_**
* Rumus **_mean_absolute_error_**

## $ \sum{|y1-  y2|}{/n}$

Dengan:
* y1 = nilai aktual pada fitur
* y2 = nilai prediksi yang dihasilkan
* n = Jumlah data


**Note:**
* Karena fitur _date_ bertipedata _datetime64[ns]_ kita ubah dulu kedalam nilai numerik, dengan menggunakan bantuan _pandas series kemudian panggil _method `map()`_ dan diisi dengan parameter `datetime.datetime.toordinal`. Ini bertujuan agar data _training_ bisa terbaca oleh model sekaligus data _validasi_, untuk hasil _outputnya_ bisa dilihat pada **gambar 13**

![hasilvoting](https://user-images.githubusercontent.com/49816104/194715361-60fe5074-d1a9-40f6-87ac-cb543141f73c.png)

**Gambar 9. Hasil _output training_ masing-masing model**

* Pada **gambar 9** nilai _loss_ yang paling kecil adalah _lasso dan linear regression_, kita bisa memilih diantara salahsatu model tersebut. Dalam kasus ini, memilih model _linear regression_ yang paling umum digunakan dan _populer dalam _machine learning_.Selanjutnya lakukan _normalisasi_ `MinMaxScaler()` pada data _training_ fungsi _normalisasi_ ini menghasilkan nilai rentan **0** sampai **1**


* Membuat _objek linear regression_, terlebih dahulu lakukan seperti biasa _normalisasi_ pada data _validasi_ bukan pada data _training_, karena data _training_ sudah _dinormalisasi_ sebelumnya

* Membuat prediksi pada model yang sudah _ditraining_, kemudian buat sebuah tabel _dataframe_ dari bawaan pandas dan simpan pada _variable_ hasil, lalu tampilkan dalam 15 baris untuk melihat perbedaan nilai aktual dan nilai yang diprediksi oleh model untuk lebih lengkapnya bisa dilihat pada **gambar 14**

|   NO  |  y_test  |  y_pred  |
|:-----:|:--------:|:--------:|
| 10140 | 357548.0 | 353820.2 |
|  7465 | 270521.0 | 407522.2 |
|  808  | 348611.0 | 360292.3 |
|  7666 | 233449.0 | 244704.6 |
|  2624 | 325025.0 | 376659.6 |
|  6448 | 224800.0 | 222025.7 |
|  5434 |  95030.0 |  41236.1 |
|  4604 | 132325.0 | 160086.5 |
| 13302 |  70612.0 | 112803.1 |
|  6708 | 126776.0 | 155219.1 |
|  878  | 531832.0 | 474328.8 |
|  3694 | 266686.0 | 148640.1 |
| 10968 | 130263.0 | 231707.0 |
|  3878 | 730646.0 | 448280.1 |
|  3842 | 656485.0 | 389657.2 |

**Tabel 5. Hasil _output_ prediksi dan aktual**

Untuk melihat hasil garis prediksi model dalam menentukan polanya, bisa dilihat menggunakan bantuan visualisasi, sekaligus dengan melihat visualisasi ini kita bisa tahu apakah model mengalami _underfitting, overfitting ataupun goodfit_.

![scatter_pred](https://user-images.githubusercontent.com/49816104/194736401-dc6ebd66-84b0-409e-82c6-114f0aed4e25.jpg)

**Gambar 10. Visualisasi hasil prediksi**
* Dari hasil prediksi model dalam menemukan polanya tampak seperti pada **gambar 10**, dengan garis prediksi berwarna **merah**. warna **biru** dan **orange** adalah wilayahnya. Dengan sudah membentuk garis prediksi kita bisa memprediksi harga rumah pada tahun yang akan datang. 


Pada tahapan ini, kita akan membuat model dalam _deep learning_. Akan tetapi terlebih dahulu membuat _object_ `EarlyStopping()` dahulu tujuannya untuk mengurangi terjadinya model _overfiting_. Dengan bantuan ini saat model sewaktu training otomatis akan berhenti, berikut beberapa _parameter_ yang digunakan:
* `monitor` = pilih model yang akan dimonitor apakah akurasi, val akurasi, _loss ataupun val loss_. Pada tahap ini kita gunakan loss karena dalam kasus kita _regressi_
* `mode` = terdiri dari pilihan `max` dan `min`, sesuai dengan pasangannya jika kita memilih `monitor` **akurasi** maka piih `max`, akan tetapi jika kita memilik `monitor` **_loss_** maka piih `min`
* `patience` = jeda dalam sewaktu _training_, maksudnya ketika model sudah mencapai _loss_ yang minim kita bisa memberi jeda _epochs_ yang ditentukan sesuai keinginan kita, dalam kasus ini kita _set_ **25**

Setelah membuat_object_ `EarlyStopping` selanjutnya kita membuat arsitektur _neural network model_ dengan membuat *2* _hidden layer_ dan _activation _set_ ke _relu_ yang paling umum digunakan, lalu pada _layer_ terakhir _set_ _activation_ ke _linear_ karena metoda pada kali ini menggunakan permasalahan _regressi_ bukan klasifikasi. Kemudian pada _compiler_ pilih _loss_ **_mean_squared_error_** pada tahapan kali ini. Dan teakhir panggil _method_ `fit` untuk melakukan training seperti biasa sedikit tambahan kita tambahkan _parameter_ `callbacks` isi dengan _object_ `EarlyStopping` yang sudah dibuat sebelumnya kemudian dirubah dalam bentuk _list_ dan pada _epochs_ kita set **1000**, untuk **arsitektur** _neural network_ yang digunakan adalah sebagai berikut:

* Menggunakan 2 _hidden layer, layer pertama sebanyak **16** neuron, activation="relu"_, dan ditambahkan _dropout_ sebanyak 0.1 random _neuron_
* _layer kedua sebanyak **16** _neuron, activation="relu"_ 
* Pada _optimizer_ gunakan yang paling umum digunakan yaitu `Adam`, dan pada _loss_ menggunakan _mean_squared_error pada kali ini.

Lalu terakhir lakukan _training_ dengan memanggil _method_ `fit()` dan tambahkan _parameter callbacks_ kemudian masukkan dengan _object_ `EarlyStopping` yang sudah dibuat kedalam bentuk _list_

**Note:**

* Untuk Rumus **_Mean Squared Error_**

## MSE = $\sum_{n}^{1}(y1-y2)^2$

Dengan :
* y1 = Nilai aktual (nilai yang sebenarnya)
* y2 = Nilai hasil prediksi model
* n = Jumlah data

* Walaupun sebelumnya sudah kita _set epochs_ pada **1000** sewaktu training akan berhenti ketika mencapai level _loss_ yang minim. Kemudian kita visualisasikan dalam bentuk _line plot_ hasil dari proses _training_ tersebut untuk melihat apakah model _underfitting, overfitting_ ataupun _goodift_ untuk lebih jelasnya bisa dilihat pada **gambar 16**

![loss](https://user-images.githubusercontent.com/49816104/194765985-90a01773-b99a-4663-853e-91ca52545353.jpg)

**Gambar 11. Hasil _training_ dalam visualisasi**

* Pada **gambar 11** model _goodfit_ karena stabil akan tetapi sedikit mulai mengalami _overfiting_ 

Selanjutnya kita prediksi model yang sudah dibuat sebelumnya lalu bandingkan dengan hasil aktual dan dibuat kedalam tabel _dataframe pandas_ supaya terlihat rapi.

|   NO  |  y_test  |  y_pred  |
|:-----:|:--------:|:--------:|
| 10140 | 357548.0 | 337066.0 |
|  7465 | 270521.0 | 386119.0 |
|  808  | 348611.0 | 342977.0 |
|  7666 | 233449.0 | 237244.0 |
|  2624 | 325025.0 | 357928.0 |
|  6448 | 224800.0 | 215053.0 |
|  5434 |  95030.0 |  43842.0 |
|  4604 | 132325.0 | 154446.0 |
| 13302 |  70612.0 | 108179.0 |
|  6708 | 126776.0 | 149683.0 |

**Tabel 5. Hasil prediksi model**

Selanjutnya kita akan melihat garis prediksi seperti biasa hasil dari _training model_ dalam menemuka pola data

![scatter_pred_2](https://user-images.githubusercontent.com/49816104/194766219-daab7802-430f-4eda-aa08-c2caccdcdd4a.png)

**Gambar 12. Visualisasi hasil prediksi**

* Berbeda pada model sebelumnya menggunakan `LinearRegression` pada **gambar  10** garis prediksi kali ini justru mengikuti pola data yang ada tidak lurus sepenuhnya, hampir mirip lengkungan _polinomial regression_

|   NO  | y_aktual |  y_pred  | y_pred_deep |
|:-----:|:--------:|:--------:|:-----------:|
| 10140 | 357548.0 | 353820.2 |   337066.0  |
|  7465 | 270521.0 | 407522.2 |   386119.0  |
|  808  | 348611.0 | 360292.3 |   342977.0  |
|  7666 | 233449.0 | 244704.6 |   237244.0  |
|  2624 | 325025.0 | 376659.6 |   357928.0  |
|  6448 | 224800.0 | 222025.7 |   215053.0  |
|  5434 |  95030.0 |  41236.1 |   43842.0   |
|  4604 | 132325.0 | 160086.5 |   154446.0  |
| 13302 |  70612.0 | 112803.1 |   108179.0  |
|  6708 | 126776.0 | 155219.1 |   149683.0  |
|  878  | 531832.0 | 474328.8 |   447143.0  |
|  3694 | 266686.0 | 148640.1 |   143246.0  |
| 10968 | 130263.0 | 231707.0 |   224526.0  |
|  3878 | 730646.0 | 448280.1 |   423349.0  |
|  3842 | 656485.0 | 389657.2 |   369800.0  |
|  5631 | 440525.0 | 361950.4 |   344492.0  |

**Tabel 6. Perbandingan hasil prediksi masing-masing model**

### **Evaluation**
---
Pada penggunaan _loss linear regression_ menggunakan **mean_absolute_error**, berbeda dengan _deep learning_ menggunakan **mean_squared_error** untuk perhitungannya adalah sebagai berikut:

**Mean Absolute Error (MAE)**
## MAE =$\frac{1}{n}\sum{|y1-  y2|}$
Dengan :
* y1 = Nilai aktual (nilai yang sebenarnya)
* y2 = Nilai hasil prediksi dari model
* n = Jumlah data

**Mean Squared Error (MSE)**
## MSE = $\frac{1}{n}\sum(y1-y2)^2$
Dengan :
* y1 = Nilai aktual 
* y2 = Nilai hasil prediksi
* n = Jumlah data

![mse](https://user-images.githubusercontent.com/49816104/194766876-9db94ed1-9cb2-43f2-9deb-2f1d29bd0491.jpg)

**Gambar 13. Perbandingan _loss_**

* Pada **gambar 13** nilai loss yang dihasilkan paling _minim_ adalah model `Linear Regression` dan `Lasso` sebenarnya cara kerja kedua model ini sama, dalam hal ini kita pilih salasatu dari kedua model ini yaitu `Linear Regression`. Model `Lasso` dan `Linear Regression` memiliki nilai yang sama, dibandingkan dengan `Random Forest` yang memiliki nilai dibawah kedua model yaitu **148.51**

* Kesimpulan

Pada tahap ini kita mempertimbangkan model-model yang sudah kita buat sebelumnya, masing-masing model memiliki kelebihan dan kekurangnnya, begitupula pada _deep learning_ untuk meningkatkan _performace_ model dengan mengubah arsitektur seperti **jumlah neuron, jumlah hidden layer, _optimizer_** dan sebagainya tetapi yang paling berpengaruh pada kasus ini adalah **jumlah neuron** semakin banyak neuron justru model akan mengalami _overfiting_. _Performace_ model untuk data ini yang paling cocok adalah metode `Linear Regression` karena memiliki nilai _error_ yang minim dibandingkan menggunakan metode _deep learning linear regression_ memiliki tingkat _error_ **147.0** sedangkan _deep learning_ **0.200**, sebernanya model ini masih bisa ditingkatkan lagi dengan menggunakan _hyperparameter tuning_ menggunakan _gridsearch_ dengan cara menambah atau mengurangi **jumlah neuron, hidden layer** dan juga pada optimizer yang digunakan terutama pada __learning_rate__

sesuai dengan _goals_ kita yaitu memprediksi harga dari tahun ke tahun dibuktikan dengan menggunakan model _linear regression_, dan untuk kepadatan rumah dari setiap tahun dibuktikan dengan visualisasi dalam bentuk **kdeplot** pada **gambar 6**, tetapi pada _performace_ model yang dibuat masih bisa ditingkatkan lagi menggunakan _deep learning_ salahsatunya teknik _hyperparameter tuning_ yang sudah disebutkan sebelumnya.
