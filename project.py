import cv2
import numpy as np

# Fungsi untuk mendeteksi dan menandai objek tangan
def deteksi_dan_penanda_tangan(frame):
    # Konversi ke citra abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi pada citra abu-abu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 100)

    # Temukan kontur tangan yang signifikan
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cari tangan dengan luas kontur terbesar
    max_area = 0
    hand_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            hand_contour = contour

    # Jika tangan terdeteksi, tandai dengan persegi
    if hand_contour is not None:
        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

# Inisialisasi video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Baca frame dari video capture
    _, frame = video_capture.read()

    # Deteksi dan penanda objek tangan
    frame_with_hand = deteksi_dan_penanda_tangan(frame)

    # Tampilkan frame yang telah ditandai
    cv2.imshow('Deteksi Tangan', frame_with_hand)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video capture dan jendela tampilan
video_capture.release()
cv2.destroyAllWindows()
