import cv2
import numpy as np
import face_recognition

imgTushar = face_recognition.load_image_file("Images/Tushar.jpg")
imgTushar = cv2.cvtColor(imgTushar,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Tushar Test.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgTushar)[0]
encodeTushar = face_recognition.face_encodings(imgTushar)[0]
cv2.rectangle(imgTushar,(faceLoc[3],faceLoc[0 ]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),4)

result = face_recognition.compare_faces([encodeTushar],encodeTest)
faceDis = face_recognition.face_distance([encodeTushar],encodeTest)
print(result,faceDis)
cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Tushar",imgTushar)
cv2.imshow("Tushar Test",imgTest)

cv2.waitKey(0)