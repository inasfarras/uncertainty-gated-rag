Siap—ini penjelasan **semua metrik** yang kamu pakai, singkat-padat tapi teknis.

---

## 1) Count (N)

* **Apa**: Jumlah pertanyaan/sampel yang dievaluasi.
* **Hitung**: Banyaknya baris pada dataset yang benar-benar dijalankan (setelah filter `--n`).
* **Baca**: Konteks ukuran sampel. N kecil → varians tinggi; hati-hati menarik kesimpulan.

## 2) Avg Faithfulness

* **Apa**: Seberapa **selaras** jawaban dengan bukti/CTX yang diberikan (0–1).
* **Hitung** (versi sekarang): fallback **`min(1.0, 0.6 + 0.4 * overlap)`**; nanti bisa diganti/ditambah **LLM-judge/RAGAS**.
* **Baca**: Mendekati 1 berarti klaim di jawaban konsisten dengan evidence. Bisa tinggi walau F1 rendah bila jawaban “benar-ke-konteks” tapi tidak sama dengan gold.

## 3) Avg Overlap

* **Apa**: Proporsi kalimat jawaban yang **benar-benar didukung** oleh potongan konteks yang disitasi.
* **Kontrak**: Setiap kalimat non-IDK boleh punya **\[CIT:\<doc\_id>]** tepat; IDK **tidak boleh** bersitasi.
* **Hitung** (per jawaban):

  1. Split kalimat; buang kalimat IDK.
  2. Ambil **doc\_id** dari sitasi; cek doc\_id valid di CTX.
  3. Hitung kemiripan semantik kalimat ↔ teks chunk milik doc\_id (cosine), **≥ τ\_sim** (mis. 0.58–0.70) ⇒ **supported**.
  4. **Overlap = (# kalimat supported) / (# kalimat klaim)**.
     Rata-ratakan ke semua sampel ⇒ **Avg Overlap**.
* **Baca**: 0.45 berarti ±45% kalimat didukung bukti yang benar.

## 4) Avg EM (Exact Match)

* **Apa**: Cocok **persis** dengan gold (setelah normalisasi string), 0 atau 1 per sampel.
* **Hitung**: Lowercase, trim spasi, buang tanda baca/artikel umum (opsional), bandingkan string prediksi vs gold; rata-ratakan.
* **Baca**: Keras/ketat; bagus untuk QA faktual pendek. Simpan di lampiran bila ruang sempit.

## 5) Avg F1

* **Apa**: Tumpang-tindih token antara jawaban dan gold (0–1), lebih “lunak” dari EM.
* **Hitung**: Tokenisasi sederhana (spasi), normalisasi seperti EM, lalu precision/recall → **F1** per sampel; rata-ratakan.
* **Baca**: Menangkap jawaban yang **sebagian benar**. Biasanya jadi metrik **task accuracy** utama.

## 6) Abstain Rate

* **Apa**: Proporsi kasus sistem **tidak menjawab** (mis. “I don’t know.” atau aksi **ABSTAIN** pada gate).
* **Hitung**: (# jawaban diklasifikasi abstain) / N.
* **Baca**: Penting untuk kebijakan gate. Tinggi ≠ buruk jika mengurangi halusinasi **dengan biaya turun**; tapi terlalu tinggi menurunkan utilitas.

## 7) Avg Total Tokens

* **Apa**: Total token LLM per pertanyaan (input + output, dijumlahkan lintas seluruh ronde/aksi).
* **Hitung**: Ambil `prompt_tokens + completion_tokens` dari metadata panggilan LLM, jumlahkan per query, lalu rata-ratakan.
* **Baca**: Proksi **biaya** (dan sering berkorelasi dengan latensi). Targetmu: turun (mis. < \~800–1200) tanpa merusak reliabilitas.

## 8) P50 Latency (ms)

* **Apa**: **Median** durasi eksekusi per pertanyaan (end-to-end untuk jalur gen), dalam milidetik.
* **Hitung**: Ambil waktu mulai-selesai per query, ambil **persentil-50**.
* **Baca**: Stabilitas kecepatan; bandingkan juga P95 di lampiran untuk ekor lambat.

## 9) IDK+Cit Count

* **Apa**: Jumlah kasus kalimat **IDK** namun **ada sitasi**.
* **Hitung**: Deteksi pola IDK (“I don’t know”, “Tidak tahu”, dll.) pada output final **yang punya \[CIT:…]**.
* **Baca**: **Harus 0** (kontrak prompt). Jika >0, artinya evaluasi Overlap/Faithfulness bisa **over-credit** dan kontrak perlu ditegakkan.

---

### Cara membaca metrik **berpasangan**

* **Faithfulness/Overlap** vs **F1/EM**:
  – Tinggi & rendah → jawaban konsisten dengan bukti namun tidak sama dengan gold (mungkin gold berbeda frasa/format).
  – Rendah & rendah → kemungkinan halusinasi/kontra-bukti **dan** salah ke gold.
* **Tokens** & **Latency** vs **Abstain Rate**:
  – Gate yang baik menurunkan **tokens/latency** sambil menaikkan **faith/overlap**; **abstain** naik secukupnya (bukan ekstrem).

### Ambang & praktik baik (rule-of-thumb, tergantung dataset)

* **IDK+Cit** = 0 (ketat).
* **τ\_sim** (Overlap) ≈ 0.6–0.7; jangan terlalu rendah agar tidak over-credit.
* **max\_output\_tokens** 128–256; **temperature=0**, **top\_p=0** untuk baseline deterministik.
* Laporkan **N, Faithfulness, Overlap, F1, Tokens, P50 Latency, Abstain** di tabel utama; **EM, P95, token breakdown** di lampiran.
