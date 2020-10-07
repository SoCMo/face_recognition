import os
import time

import MyThread

filePath = "Image//"
MyThread.myThread.filePath = filePath

now = 1

allImage = os.listdir(filePath)
for image in allImage:
    if image.endswith((".png", ".jpg")):
        thread = MyThread.myThread(str(now), image, now)
        thread.start()
        now = now + 1
        time.sleep(0.05)
