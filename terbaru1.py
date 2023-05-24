import cv2
import torch

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Set device to CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Load image
image_path = 'coba2.jpg'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Filter the results to only include hands
hands_results = results.pandas().xyxy[results.pandas().xyxy['name'] == 'hand']

# Draw bounding boxes around hands
for _, row in hands_results.iterrows():
    xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, 'Hand', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Hands Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
