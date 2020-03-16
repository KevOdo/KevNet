import cv2 as cv
import os
import csv

top = './UCF11/'
path = 'frames/'
endin = '01.mpg'
file_paths = []

classes = [ 'basketball',
            'biking',
            'diving',
            'golf',
            'riding',
            'juggle',
            'swing',
            'tennis',
            'jumping',
            'spiking',
            'walk',
]

for r,d,f in os.walk(top, topdown=True):
  for file in f:
    if endin in file:
      file_paths.append("{}/{}".format(r, file))

with open('frames.csv', mode='w') as csv_file:
  writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['Frames', 'Class'])

  for file in file_paths:
    cap = cv.VideoCapture(file)
    if (cap.isOpened() == False):
      print('Error opening video')

    fcount = 0
    frames = []
    vid_cat = ''
    for cat in classes:
      if cat in file:
        vid_cat = cat

    while cap.isOpened():
      ret, frame = cap.read()
      fcount += 1
      if ret == True:
        if fcount % 5 == 0:
          frames.append(frame)

      else:
        break
    
    cap.release()
    cv.destroyAllWindows()

    fstr = ''
    for i, frame in enumerate(frames):
      filename = (file[1:-4]+'_'+str(i)+'.png').replace('/','_')
      cv.imwrite(path + filename, frame)
      fstr += filename + '|'

    writer.writerow([fstr, vid_cat])