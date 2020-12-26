import cv2
import matplotlib.pyplot as plt

lotr = cv2.imread("lotr.jpg")
lotr = cv2.cvtColor(lotr, cv2.COLOR_BGR2RGB)

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rectangle = cascade.detectMultiScale(lotr, minNeighbors = 5)

for (x,y,w,h) in face_rectangle:
    
    cv2.rectangle(lotr,(x,y), (x + w, y + h), (0,255,0), 10)
    plt.title("LOTR Face Detection")
    plt.axis("off")
    plt.imshow(lotr)
    
if 0xFF == ord("q"):
    cv2.destroyAllWindows()
