import threading

import cv2
import face_recognition
import numpy as np


class FaceThread(threading.Thread):
    threadLock = threading.Lock()
    known_face_encodings = []
    known_face_names = []

    face_names_class = []
    face_encodings_class = []
    face_locations_class = []

    def __init__(self, threadID, frame):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.frame = frame

    def run(self):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        print("%s:face_recognition Done" % self.threadID)
        FaceThread.threadLock.acquire()
        FaceThread.face_names_class = face_names
        FaceThread.face_encodings_class = face_encodings
        FaceThread.face_locations_class = face_locations
        print("%s:store Done" % self.threadID)
        FaceThread.threadLock.release()
