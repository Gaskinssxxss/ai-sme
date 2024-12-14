
# **Analisis Preferensi Pemilih Pilkada Kota Bima**

Proyek ini bertujuan untuk menganalisis preferensi pemilih terhadap kandidat Pilkada Kota Bima berdasarkan data responden yang berasal dari berbagai kecamatan. Analisis ini melibatkan pengolahan data, visualisasi, dan prediksi menggunakan algoritma **K-Nearest Neighbors (KNN)**.

---

## **Struktur Proyek**
### **1. File dan Dataset**
- **DATA.xlsx**: Dataset utama berisi informasi responden, termasuk kecamatan dan kandidat pilihan mereka.
- **bootstrap_kec.xlsx**: Dataset hasil resampling menggunakan metode bootstrap sebanyak 8100 sampel.
- **encode_datax.xlsx**: Dataset hasil encoding variabel independen (`Kecamatan`) dan variabel dependen (`Kandidat`) menjadi format numerik.

### **2. Output Visualisasi**
- **Persentase Responden per Kecamatan**:
  - Diagram batang yang menunjukkan distribusi responden berdasarkan kecamatan.
  - File: `Persentase_Responden_Kecamatan.jpeg`
- **Persentase Responden per Kandidat**:
  - Diagram batang yang menunjukkan proporsi kandidat yang dipilih oleh responden.
  - File: `Persentase_Responden_Kandidat.jpeg`
- **Distribusi Prediksi Kandidat Berdasarkan Kecamatan**:
  - Diagram batang terkelompok yang menunjukkan prediksi kandidat berdasarkan kecamatan.
  - File: `Distribusi_Prediksi_Kandidat.jpeg`

### **3. Kode Python**
Kode Python digunakan untuk:
1. Membaca dataset, melakukan preprocessing, dan resampling.
2. Melakukan encoding data untuk model prediksi.
3. Membangun model prediksi menggunakan algoritma **K-Nearest Neighbors (KNN)**.
4. Membuat visualisasi data menggunakan **Matplotlib** dan **Seaborn**.

---

## **Langkah Analisis**
### **1. Data Preparation**
- **Load Dataset**: Data diambil dari file `DATA.xlsx` menggunakan pustaka `pandas`.
- **Filtering & Resampling**: Data difilter untuk menghapus nilai kosong, kemudian dilakukan resampling menggunakan bootstrap.

### **2. Visualisasi**
- **Distribusi Responden Berdasarkan Kecamatan**:
  - Menggunakan diagram batang untuk melihat distribusi responden per kecamatan.
- **Distribusi Kandidat yang Dipilih Responden**:
  - Menggunakan diagram batang untuk menampilkan preferensi kandidat secara keseluruhan.
- **Distribusi Prediksi Kandidat Berdasarkan Kecamatan**:
  - Menggunakan diagram batang terkelompok berdasarkan hasil prediksi model KNN.

### **3. Encoding Data**
- **LabelEncoder** digunakan untuk mengubah data kategorikal (kecamatan dan kandidat) menjadi format numerik agar dapat digunakan oleh model KNN.

### **4. Prediksi dengan KNN**
- Data dibagi menjadi **70% data latih** dan **30% data uji**.
- Model KNN dilatih dengan parameter `k=5`.
- Hasil prediksi diverifikasi menggunakan data uji, dan akurasi model dihitung.

---

## **Detail Kode**
### **1. Membaca Dataset**
```python
import pandas as pd

# Membaca dataset utama
df_data = pd.read_excel('DATA.xlsx', engine='openpyxl')
df_data.head()
```

### **2. Resampling Data**
```python
from sklearn.utils import resample
import os

filterisasi = df_data[['Kecamatan', 'Kandidat']].dropna()
resampling = resample(filterisasi, replace=True, n_samples=8100, random_state=42)

# Menyimpan hasil resampling
direktori = "Resamplig"
os.makedirs(direktori, exist_ok=True)
file = os.path.join(direktori, "bootstrap_kec.xlsx")
resampling.to_excel(file, index=False)
```

### **3. Visualisasi Data**
**Persentase Responden per Kecamatan**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

total_kecamatan = resampling['Kecamatan'].value_counts()
persentase_kec = (total_kecamatan / len(resampling)) * 100

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=persentase_kec.index, y=persentase_kec.values, color="#1f77b4")

plt.title('Persentase Responden Tiap Kecamatan', fontsize=16)
plt.xlabel('Kecamatan', fontsize=12)
plt.ylabel('Persentase (%)', fontsize=12)

for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{p.get_height():.2f}%', 
            ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.show()
```

### **4. Encoding Data**
```python
from sklearn.preprocessing import LabelEncoder

label_encoding = LabelEncoder()
resampling['Kecamatan_encoded'] = label_encoding.fit_transform(resampling['Kecamatan'])
resampling['Kandidat_encoded'] = label_encoding.fit_transform(resampling['Kandidat'])

# Menyimpan hasil encoding
file = 'Encode/encode_datax.xlsx'
resampling.to_excel(file, index=False)
```

### **5. Prediksi dengan KNN**
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = resampling[['Kecamatan_encoded']]
y = resampling['Kandidat_encoded']

# Membagi data latih dan uji
kec_train, kec_testing, kandidat_train, kandidat_testing = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(kec_train, kandidat_train)

# Evaluasi model
accuracy = knn.score(kec_testing, kandidat_testing)
print(f"Accuracy of the KNN model with k={k}: {accuracy:.2f}")
```

### **6. Visualisasi Prediksi**
```python
prediksi = pd.DataFrame({
    'Kecamatan_encoded': kec_testing['Kecamatan_encoded'],
    'Predicted_Kandidat': knn.predict(kec_testing)
})

# Visualisasi distribusi prediksi
distribusi_kandidat = prediksi.groupby(['Kecamatan_encoded', 'Predicted_Kandidat']).size().unstack(fill_value=0)
ax = distribusi_kandidat.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')

plt.title('Distribusi Prediksi Kandidat Berdasarkan Kecamatan', fontsize=14)
plt.xlabel('Kecamatan', fontsize=12)
plt.ylabel('Jumlah Prediksi', fontsize=12)
plt.legend(title='Kandidat', bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

---

## **Hasil**
1. **Akurasi Model KNN**:
   - Akurasi model dengan parameter `k=5` adalah **68%**.

2. **Distribusi Responden**:
   - Kecamatan dengan jumlah responden tertinggi adalah **Raba (25,43%)**.
   - Kandidat yang paling banyak dipilih adalah **H. Arrahman Abidin (43,33%)**.

3. **Hasil Prediksi**:
   - Hasil prediksi menunjukkan distribusi kandidat berbeda di setiap kecamatan, dengan hasil yang mendukung kandidat tertentu di wilayah-wilayah tertentu.

---

## **Kesimpulan**
- **Analisis Data**: Menunjukkan distribusi pemilih dan kandidat favorit berdasarkan data.
- **Model KNN**: Mampu memprediksi kandidat pilihan responden berdasarkan kecamatan dengan akurasi yang cukup baik.
- **Visualisasi**: Membantu memahami pola distribusi pemilih dan hasil prediksi kandidat.
