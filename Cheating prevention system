import cv2
import time

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to keep track of face detection
last_detected_time = time.time()
alert_no_face_interval = 10  # seconds to wait before alerting for no face detected
alert_multiple_faces = False


# Function to detect faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        faces = detect_faces(frame)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Alert if no face detected for a certain interval
        current_time = time.time()
        if len(faces) == 0:
            if current_time - last_detected_time > alert_no_face_interval:
                print("Alert: No face detected for 10 seconds!")
        else:
            last_detected_time = current_time

        # Alert if multiple faces are detected
        if len(faces) > 1 and not alert_multiple_faces:
            print("Alert: Multiple faces detected!")
            alert_multiple_faces = True
        elif len(faces) <= 1:
            alert_multiple_faces = False

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting the program.")

finally:
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
