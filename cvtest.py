import cv2
import sys
from datetime import datetime, time

mVideoPath = r"P:\UT\2019-07-18_00-49-02_Slabo.mp4" #Not read 
mVideoPath = r"C:\Video\2019-07-18_00-45-13_Slabo.mkv"
#mVideoPath = "P:\\UT\\2019-07-18_00-49-02_Slabo.mp4" //NOT
#mVideoPath = r"C:\Video\Vulpev 2019 18_20mbps.avi" #OK
#mVideoPath = r"C:\Video\arnaud13_nadpisi_-1024809718.avi" #OK
cap = cv2.VideoCapture(mVideoPath)
print(cap)
print(dir(cap))
ret, frame = cap.read(0)
while(ret):
  if ret: cv2.imshow("cap", frame)     
  else: print(str(ret), str(frame))  
  cv2.waitKey(5)
  ret, frame = cap.read(0)
print("Enter...")
cap.release()
s = input()
#cv2.waitKey(0p)

