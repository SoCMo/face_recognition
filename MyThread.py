import threading

import face_recognition
import pymysql


class myThread(threading.Thread):
    filePath = ""
    sql = "INSERT INTO `face_recognition_dev`.`faceset` (`name`, `encoding`) VALUES (%s, %s);"
    threadLock = threading.Lock()

    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        find_image = face_recognition.load_image_file(self.filePath + self.name)
        encodings = face_recognition.face_encodings(find_image, model="cnn")
        if len(encodings) <= 0:
            return 0

        name = "UnKnow"
        if self.name.endswith(".jpg"):
            name = self.name.replace(".jpg", "")
        else:
            name = self.name.replace(".png", "")
        encoding = encodings[0].tostring()

        db = pymysql.connect("songcm.cn", "root", "D2018shu!", "face_recognition")
        try:
            cursor = db.cursor()
            cursor.execute(self.sql, (name, encoding))
            db.commit()

            self.threadLock.acquire()
            print("已经处理完第%s个文件:%s" % (self.counter, name))
            self.threadLock.release()
        except Exception as e:
            print(e)
            db.rollback()
        finally:
            db.close()
