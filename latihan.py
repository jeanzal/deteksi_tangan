import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model CNN untuk deteksi tangan
model = load_model('model_deteksi_tangan.h5')

# Fungsi untuk mendeteksi dan menandai objek tangan pada frame menggunakan model CNN
def deteksi_dan_penanda_tangan(frame):
    # Ubah frame menjadi citra abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ubah ukuran citra menjadi 64x64 (sesuaikan dengan ukuran input model CNN)
    resized = cv2.resize(gray, (64, 64))

    # Normalisasi nilai piksel
    normalized = resized / 255.0

    # Ubah dimensi citra menjadi bentuk yang diterima oleh model CNN (batch, tinggi, lebar, saluran warna)
    input_data = np.expand_dims(normalized, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)

    # Prediksi menggunakan model CNN
    predictions = model.predict(input_data)

    # Ambil label prediksi dengan probabilitas tertinggi
    predicted_label = np.argmax(predictions[0])

    # Jika label prediksi adalah tangan (misalnya, 1), tandai objek tangan pada frame dengan kotak pembatas
    if predicted_label == 1:
        x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, predicted_label


# Fungsi untuk merangkum hasil deteksi tangan dan mencatat jumlah tangan yang terdeteksi
def rangkum_deteksi(jenis_deteksi, jumlah_tangan):
    with open('hasil_deteksi.txt', 'a') as file:
        file.write(f"Deteksi: {jenis_deteksi}, Jumlah Tangan: {jumlah_tangan}\n")

    print(f"Deteksi: {jenis_deteksi}, Jumlah Tangan: {jumlah_tangan}")

# Fungsi untuk memilih gambar dari lokal, mendeteksi tangan pada gambar tersebut, dan menampilkan hasil deteksi
def pilih_gambar_lokal():
    image_path = input("Masukkan path lengkap file gambar: ")
    image = cv2.imread(image_path)
    image_with_hand, predicted_label = deteksi_dan_penanda_tangan(image)
    rangkum_deteksi("Gambar", predicted_label)
    cv2.imshow('Deteksi Tangan - Gambar', image_with_hand)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Fungsi untuk memilih video dari lokal, mendeteksi tangan pada setiap frame video, mencatat total tangan yang terdeteksi, dan menampilkan hasil deteksi
def pilih_video_lokal():
    video_path = input("Masukkan path lengkap file video: ")
    video_capture = cv2.VideoCapture(video_path)
    total_tangan_video = 0
    frame_video = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_with_hand, predicted_label = deteksi_dan_penanda_tangan(frame)
        total_tangan_video += predicted_label
        frame_video += 1

        rangkum_deteksi("Video", predicted_label)

        cv2.imshow('Deteksi Tangan - Video', frame_with_hand)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    print(f"Total tangan yang terdeteksi pada video: {total_tangan_video}")

# Fungsi untuk menggunakan kamera live, mendeteksi tangan pada setiap frame video yang diambil dari kamera, mencatat hasil deteksi, dan menampilkan hasil deteksi
def kamera_live():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        frame_with_hand, predicted_label = deteksi_dan_penanda_tangan(frame)

        rangkum_deteksi("Kamera Live", predicted_label)

        cv2.imshow('Deteksi Tangan - Kamera Live', frame_with_hand)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    print("Menu:")
    print("1. Pilih Gambar Lokal")
    print("2. Pilih Video Lokal")
    print("3. Kamera Live")

    pilihan = input("Pilih menu (1/2/3): ")

    if pilihan == "1":
        pilih_gambar_lokal()
    elif pilihan == "2":
        pilih_video_lokal()
    elif pilihan == "3":
        kamera_live()
    else:
        print("Pilihan tidak valid.")

