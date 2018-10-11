# dependencies
import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# use tqdm to seee progress bar
from tqdm import tqdm
sampleNum=0;
while 1:
    ret, img = tqdm(cap.read())
    no=sampleNum
    print("image_no :",no)
    # convert image color from RGB to GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sampleNum+=1
        cv2.imwrite("dataSet1/shivam_img1/pshivam."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
    cv2.imshow("face",img)
    # will wait 5 milli second   
    cv2.waitKey(5)
    # will take 10 photos
    if(sampleNum>10):
        break

# release open camara
cap.release()
# destroy all open windows
cv2.destroyAllWindows()
