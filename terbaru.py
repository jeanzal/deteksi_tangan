import cv2
import torch

# Langkah 2: Muat Model YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Langkah 4: Mendeteksi dan Menandai Objek pada Gambar
def detect_and_mark_objects(image_path):
    # Baca gambar dari file
    image = cv2.imread(image_path)

    # Lakukan deteksi objek menggunakan model YOLO
    results = model(image)

    # Tampilkan hasil deteksi dan penandaan pada gambar
    results.show()

# Langkah 5: Mendeteksi dan Menandai Objek dalam Video
def detect_and_mark_objects_in_video(video_path):
    # Baca video dari file
    video = cv2.VideoCapture(video_path)

    while True:
        # Baca frame dari video
        ret, frame = video.read()

        if not ret:
            break

        # Lakukan deteksi objek menggunakan model YOLO pada setiap frame
        results = model(frame)

        # Tampilkan hasil deteksi dan penandaan pada setiap frame
        results.show()

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Langkah 6: Mendeteksi dan Menandai Objek dalam Kamera Live
def detect_and_mark_objects_in_live_camera():
    # Buka kamera menggunakan OpenCV
    camera = cv2.VideoCapture(0)

    while True:
        # Baca frame dari kamera
        ret, frame = camera.read()

        # Lakukan deteksi objek menggunakan model YOLO pada setiap frame
        results = model(frame)

        # Tampilkan hasil deteksi dan penandaan pada setiap frame
        results.show()

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Contoh Penggunaan
image_path = 'coba.jpg'
video_path = 'deteksi_video.mp4'

# Mendeteksi dan menandai objek pada gambar
detect_and_mark_objects(image_path)

# Mendeteksi dan menandai objek dalam video
detect_and_mark_objects_in_video(video_path)

# Mendeteksi dan menandai objek dalam kamera live
detect_and_mark_objects_in_live_camera()
