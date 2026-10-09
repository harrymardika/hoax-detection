# Hoax Detection: Pipeline MLOps dengan TFX

Pipeline machine learning end-to-end untuk mendeteksi berita hoaks berbahasa Indonesia, dibangun dengan **TensorFlow Extended (TFX)**: validasi data, hyperparameter tuning, pelatihan CNN teks, evaluasi dengan TFMA, deployment dengan **TensorFlow Serving** di Railway, dan monitoring dengan **Prometheus + Grafana**.

Proyek ini adalah submission kelas MLOps Dicoding (*Submission 1: Hoax Detection*). **Nama:** Harry Mardika · **Username Dicoding:** hkacode

## Ringkasan Hasil

| Metrik | Nilai |
|---|---|
| Accuracy / loss (train) | 0.8961 / 0.2798 |
| Validation accuracy / loss | 0.8118 / 0.6459 |
| Hyperparameter terbaik | vocab 14.000, panjang sekuens 125, embedding 128, Conv1D 128 filter, Dense 64 dan 80, dropout 0.2 dan 0.3, learning rate ≈ 7e-5 |
| Tuning | Keras Tuner RandomSearch, 30 trial, objective `val_loss` (terbaik 0.4569) |

Model yang disimpan berasal dari epoch pertama (val_loss terendah). Setelah itu akurasi training naik ke 99,9% tetapi val_loss terus naik, tanda overfitting cepat.

**Catatan:** dataset tidak seimbang (sekitar 82% berlabel hoaks), sehingga akurasi validasi 81% hampir sama dengan model yang selalu menebak "hoaks". Precision, recall, dan AUC dari Evaluator lebih tepat untuk menilai model. Penanganan ketidakseimbangan kelas (class weight atau resampling) adalah langkah perbaikan berikutnya.

## Deskripsi Proyek

| | **Deskripsi** |
| --- | --- |
| **Dataset** | [Indonesia False News (Hoax) Dataset](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews/data): 4.231 narasi berita (`narasi`) berlabel `1` = hoaks (3.465) dan `0` = bukan hoaks (766). |
| **Masalah** | Hoaks dapat menyebabkan ketidakpercayaan publik, memengaruhi opini dan keputusan, bahkan memicu konflik sosial. Tantangannya: keberagaman gaya penulisan, variasi topik, dan strategi manipulatif pembuat berita palsu. |
| **Solusi** | Model deep learning CNN yang mengenali pola umum berita palsu (kata emosional, judul sensasional, konten tidak kredibel) dari urutan kata dalam kalimat. |
| **Pengolahan** | Di komponen Transform: lowercase, hapus tanda kutip, hapus karakter khusus dengan regex, rapikan spasi berlebih, dan pangkas spasi di awal/akhir. |
| **Arsitektur** | Input string → TextVectorization → Embedding → Conv1D → GlobalMaxPooling1D → Dense + Dropout → Dense + Dropout → Dense sigmoid (0 = bukan hoaks, 1 = hoaks). |
| **Metrik** | Pelatihan memakai binary crossentropy dan accuracy. Evaluator (TFMA) juga menghitung AUC, precision, recall, dan example count; model di-*bless* jika binary accuracy ≥ 0.5 dan tidak lebih buruk dari baseline. |
| **Deployment** | TensorFlow Serving di Railway. Endpoint pernah di `hoax-model-production.up.railway.app`, kini sudah tidak aktif; model bisa dijalankan lokal dengan Docker (lihat bawah). |
| **Monitoring** | Prometheus dan Grafana memantau metrik TF Serving: `tensorflow:core:saved_model:read` (pembacaan SavedModel), `tensorflow:core` (eksekusi graph), `tensorflow:cc:saved_model` (waktu memuat model), dan `tensorflow:serving` (jumlah request). |

## Arsitektur Pipeline

```
data/Data_latih.csv
  → CsvExampleGen (train:eval = 8:2)
  → StatisticsGen → SchemaGen → ExampleValidator
  → Transform (pembersihan teks)
  → Tuner (30 trial) → Trainer (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  → Resolver (latest blessed) + Evaluator (TFMA)
  → Pusher → outputs/serving_model
  → TensorFlow Serving (Docker, Railway) → Prometheus → Grafana
```

Pipeline dijalankan lokal dengan Apache Beam (`BeamDagRunner`).

## Tech Stack

TFX, TensorFlow Transform, TensorFlow Model Analysis, Keras Tuner, TensorFlow/Keras, Apache Beam, TensorFlow Serving, Docker, Railway, Prometheus, Grafana (Python 3.9).

## Struktur Proyek

```
hoax-detection/
├── data/Data_latih.csv
├── modules/                  # components, pipeline, transform, tuner, model, trainer
├── hkacode-pipeline.ipynb    # Menjalankan pipeline end-to-end
├── hkacode-testing.ipynb     # Request prediksi ke model yang di-deploy
├── outputs/                  # Artefak komponen TFX dan serving_model
├── monitoring/               # Dockerfile Prometheus, prometheus.yml, prometheus.config
└── Dockerfile                # Image TensorFlow Serving
```

## Cara Menjalankan

```bash
git clone https://github.com/harrymardika/hoax-detection.git
cd hoax-detection
conda create -n mlops-tfx python=3.9 && conda activate mlops-tfx
pip install tfx
jupyter notebook hkacode-pipeline.ipynb      # jalankan pipeline
```

Serving lokal:

```bash
docker build -t hoax-model . && docker run -p 8501:8501 hoax-model
curl http://localhost:8501/v1/models/hoax-model
```

Model menerima `tf.train.Example` ter-serialisasi dan di-encode base64 (contoh lengkap di `hkacode-testing.ipynb`):

```python
import base64, requests, tensorflow as tf

def predict(text: bytes):
    feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
    ex = tf.train.Example(features=tf.train.Features(feature={"narasi": feat}))
    payload = {"signature_name": "serving_default",
               "instances": [{"examples": {"b64": base64.b64encode(ex.SerializeToString()).decode()}}]}
    return requests.post("http://localhost:8501/v1/models/hoax-model:predict", json=payload).json()

print(predict(b"Presiden Jokowi mengadakan kunjungan kerja"))   # skor > 0.5 berarti hoaks
```

Monitoring: ubah target di `monitoring/prometheus.yml` ke alamat server model, lalu `cd monitoring && docker build -t hoax-monitoring . && docker run -p 9090:9090 hoax-monitoring`, dan hubungkan Grafana ke `http://localhost:9090`.

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
