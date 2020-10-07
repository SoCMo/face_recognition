# Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.

import cv2
import numpy as np
import pymysql

from FaceThread import FaceThread

video_capture = cv2.VideoCapture(0)

db = pymysql.connect("songcm.cn", "root", "D2018shu!", "face_recognition")
print("载入中!", end=" ")

try:
    cursor = db.cursor()
    cursor.execute("SELECT count(*) FROM `faceset`;")
    sum = cursor.fetchone()[0]

    for num in range(sum // 1000 + 1):
        if num != 0:
            cursor.execute("SELECT * FROM `faceset` LIMIT " + str(num) + "000, 1000;")
        else:
            cursor.execute("SELECT * FROM `faceset` LIMIT 0, 1000")
        results = cursor.fetchall()
        for result in results:
            FaceThread.known_face_names.append(result[1])
            ndarray = np.frombuffer(result[2], dtype=np.float)
            FaceThread.known_face_encodings.append(ndarray)
        print("\r载入中!....%.1f" % ((num + 1) / (sum // 1000 + 1) * 100) + "%", end=" ")

except Exception as e:
    print(e)
    db.rollback()
finally:
    db.close()
    print()


# Initialize some variables
process_this_frame = 0
cv2.namedWindow("Video")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    print("%s:get the frame" % process_this_frame)

    if process_this_frame % 100 == 0:
        faceThreadNew = FaceThread(process_this_frame, frame)
        faceThreadNew.start()
    process_this_frame = process_this_frame + 1

    print("len of face_locations_class: %d, len of face_names_class: %d" % (
            len(FaceThread.face_locations_class),
            len(FaceThread.face_names_class)
    ))
    for (top, right, bottom, left), name in zip(FaceThread.face_locations_class,
                                                FaceThread.face_names_class):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
