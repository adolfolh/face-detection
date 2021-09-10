import cv2

# Load some pre-trained data on face frontals from opencv (using haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get web cam feed
webcam = cv2.VideoCapture(0)

# Run loop through all frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to greyscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces with trained algorithm
    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Display the images with faces & rectangles
    cv2.imshow("Face Detection (Press Q to Quit)", frame)

    # Window waits for 'Q' key to be pressed
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# Release webcam object
webcam.release()

print("Code Completed")