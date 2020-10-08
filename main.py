# Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.

import cv2
import face_recognition
import numpy as np
import pymysql


def distance(faceInfo1, faceInfo2):
    x1 = faceInfo1[1] - faceInfo1[3]
    x2 = faceInfo2[1] - faceInfo2[3]
    y1 = faceInfo1[0] - faceInfo1[2]
    y2 = faceInfo2[0] - faceInfo2[2]
    return pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)


known_face_encodings = []
known_face_names = []
face_names = []
face_locations_class = []
video_capture = cv2.VideoCapture(0)

db = pymysql.connect("songcm.cn", "root", "D2018shu!", "face_recognition_dev")
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
            known_face_names.append(result[1])
            ndarray = np.frombuffer(result[2], dtype=np.float)
            known_face_encodings.append(ndarray)
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
    # print("%s:get the frame" % process_this_frame)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame, 3)

    if process_this_frame % 12 == 0 or (len(face_names) == 0 and process_this_frame % 2 == 0):
        face_locations_class = face_locations
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    if len(face_locations) != 0 and process_this_frame % 15 != 0:
        for face_location in face_locations:
            index = -1
            minDistance = 9999999
            for nowKnowIndex, knowFace in enumerate(face_locations_class):
                temp = distance(face_location, knowFace)
                if temp < minDistance:
                    minDistance = temp
                    index = nowKnowIndex
            if index != -1:
                face_locations_class[index] = face_location

    for (top, right, bottom, left), name in zip(face_locations_class, face_names):
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
    process_this_frame = process_this_frame + 1

video_capture.release()
cv2.destroyAllWindows()
