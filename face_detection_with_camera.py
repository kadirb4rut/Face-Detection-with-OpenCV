import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while cap.isOpened():
    
    ret, frame = cap.read()
    rectangle = face_cascade.detectMultiScale(frame, minNeighbors = 10) 
    
    for (x,y,w,h) in rectangle:
        cv2.rectangle(frame,(x,y), (x + w, y + h), (0,255,0), 10)
        
    frame = cv2.flip(frame, 1)
    cv2.imshow("Face Detection", frame)
        
    if(cv2.waitKey(1)) & 0xFF == ord("q"):
        break 
    
cap.release()
cv2.destroyAllWindows()
            