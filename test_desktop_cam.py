import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

if not cap. isOpened():
    print(f"Error: Could not open camera")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 24.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab a frame.")
        break

    results = model(frame)

    cv2.imshow('Inferred image', results[0].plot())
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Release resources and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
