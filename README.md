## Review Prediction Model for Movie Review

#### by Wingstop Canduers

#### Samuel Tupa Febrian 	 – 1606878713
#### Ryan Naufal P. 		 – 1606884716
#### M. Fakhrillah Abdul Azis – 1606917531

### Description
Proyek ini akan membuat sebuah algoritma yang akan memprediksi rating dari 
sebuah review movie. Alurnya secara umum, yaitu akan diinput training data 
berupa review yang sudah di-rating untuk membuat model, yang nantinya akan 
dicoba dengan test data berupa review belum di-rating.

### Progress 1
- Setidaknya dapat melakukan eksperimen dengan file ipynb
- Logistic Regression untuk progress ini menggunakan Library sklearn
- Perubahan library Sentiment Scoring. Dari SentiWord menjadi nltk vader
- Dataset: 27500 review dengan rating 0-10. Dibagi menjadi train dan test
- Rasio Test:Train = 1:4 (Test size = 0.2)
- Menggunakan K-Fold dari library sklearn
####Hasil Eksperimen
- Akurasi
  - Tanpa KFold : 0.25983333333333336
  - Dengan KFold k=1000: 0.25983333333333336

### Progress 2
- Mengubah metode menjadi Hybrid ANN-NB
- Menggunakan CountVectorizer unigram untuk feature extraction
- Dataset: 540 movie review dengan rating 1-10. Hasil data crawling.
- Reduksi class menjadi 5, dengan menggabungkan 2 rating yang berdekatan.
- Classes:
  - 1 to 2
  - 3 to 4
  - 5 to 6
  - 7 to 8
  - 9 to 10
- Metrics:
  - Akurasi Prediksi
  - Learning Curve, khusus Hybrid ANN-NB

### Final Progress
- Feature extraction ditambah dengan proses stemming sebelum CountVectorizer
- Dataset ditambah menjadi 1885 review
#### Hasil Eksperimen
- Akurasi hasil CV=10
  - Max : 0.4830508474576271
  - Min : 0.4046610169491525
  - Mean : 0.4398305084745762
  
### Resources
- Dataset dicrawling dengan Scrapper.ipynb

### Libraries
- nltk
- sklearn
- keras
- tensorflow