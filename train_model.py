import cv2
import sys

face_type = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
img_counter = 1

"""
path = os.path.join(self.face_dir, self.face_name)
if not os.path.isdir(self.path):
    os.mkdir(self.path)
"""

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if img_counter <= 20:

            # capture image of face in grayscale 
            img_name = "{}_{}.png".format(face_type, img_counter)
            small = frame[y:y+h, x:x+w]
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(img_name, small)
            print("{} written!".format(img_name))
            img_counter += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, face_type, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
            
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



"""
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# create eigenface model
model = cv2.face.EigenFaceRecognizer_create()
images = []
labels = []

# train eigenface model
model.train(images, lagels)
model.save('trained_data_for_eigen.xml')
print("Training Complete.")
"""

