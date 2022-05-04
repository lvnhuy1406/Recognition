import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = "IMG"
images = []
classNames = []
listImg = os.listdir(path)
print(listImg)

for cl in listImg:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(len(images))
print(classNames)

def encoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = encoding(images)
print("Successful encryption!")
print(len(encodeListKnow))

def joinNow(name):
    with open("JoinNow.csv", "r+") as file:
        myDataList = file.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            file.writelines(f"\n{name}, {dtString}")

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frameSize = cv2.resize(frame, (0, 0), None, fx=0.25, fy=0.25)
    frameSize = cv2.cvtColor(frameSize, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(frameSize)
    encodeCurFrame = face_recognition.face_encodings(frameSize, faceCurFrame)

    for encodeFace, faceLocat in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            joinNow(name)

        else:
            name = 'Unknown'

        y1, x2, y2, x1 = faceLocat
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)




    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()