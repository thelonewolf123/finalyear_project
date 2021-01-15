import cv2
import face_recognition

cap = cv2.VideoCapture('/home/lonewolf/Videos/Webcam/2020-12-24-091144.webm')

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("cascade/frontalFace10/haarcascade_frontalface_alt2.xml")
# humanCasecade = cv2.CascadeClassifier("cascade/body10/haarcascade_upperbody.xml")
padding = 5

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    # humans = humanCasecade.detectMultiScale(gray,1.3,5)
    # faces= face_recognition.face_locations(gray)
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x+padding, y+padding),
                      (x+w+padding, y+h+padding), (0, 255, 0), 2)

    # for (x, y, w, h) in humans:
    #     cv2.rectangle(frame, (x+padding, y+padding),
    #                   (x+w+padding, y+h+padding), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
