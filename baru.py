import cv2
import numpy as np

# Fungsi untuk mendeteksi dan menandai objek tangan pada frame
def deteksi_dan_penanda_tangan(frame):
    # Ubah frame menjadi citra abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lakukan operasi thresholding atau operasi lain yang sesuai untuk mendapatkan gambar biner dengan tangan yang terdeteksi
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cari kontur pada gambar biner
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tandai objek tangan pada frame dengan kotak pembatas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, len(contours)


# Fungsi untuk merangkum hasil deteksi tangan dan mencatat jumlah tangan yang terdeteksi
def rangkum_deteksi(jenis_deteksi, jumlah_tangan):
    with open('hasil_deteksi.txt', 'a') as file:
        file.write(f"Deteksi: {jenis_deteksi}, Jumlah Tangan: {jumlah_tangan}\n")

    print(f"Deteksi: {jenis_deteksi}, Jumlah Tangan: {jumlah_tangan}")

# Fungsi untuk memilih gambar dari lokal, mendeteksi tangan pada gambar tersebut, dan menampilkan hasil deteksi
def pilih_gambar_lokal():
    image_path = input("Masukkan path lengkap file gambar: ")
    image = cv2.imread(image_path)
    image_with_hand, jumlah_tangan_gambar = deteksi_dan_penanda_tangan(image)
    rangkum_deteksi("Gambar", jumlah_tangan_gambar)
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

        frame_with_hand, jumlah_tangan = deteksi_dan_penanda_tangan(frame)
        total_tangan_video += jumlah_tangan
        frame_video += 1

        rangkum_deteksi("Video", jumlah_tangan)

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

        frame_with_hand, jumlah_tangan = deteksi_dan_penanda_tangan(frame)

        rangkum_deteksi("Kamera Live", jumlah_tangan)

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

