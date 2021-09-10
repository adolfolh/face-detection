import cv2

# Load some pre-trained data on face frontals from opencv (using haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect the faces in
img = cv2.imread('img.jpeg')

# Convert to greyscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces with trained algorithm
face_coordinates = trained_face_data.detectMultiScale(gray_img)

# Draw rectangles around faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 4)

# Display the images with faces & rectangles
cv2.imshow("Face Detection", img)

# Window waits for key to be pressed
cv2.waitKey()


print("Code Completed")