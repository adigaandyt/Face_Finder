from datetime import timedelta
import cv2
import numpy as np
import os
import face_recognition as fr
import shutil
import sys
from os.path import exists
from simple_facerec import SimpleFacerec
sfr = SimpleFacerec()
"""
Extract a frame every 5 seconds from the videos in the Vids folder
and save them in the Frames folder
"""


def extract_frames():
    dir = r'./Workspace/Vids/'
    pathOut = r"./Workspace/Frames/"
    count = 0
    counter = 1
    listing = os.listdir(dir)
    for vid in listing:
        print("Extracting " + vid)
        vid = dir + vid
        cap = cv2.VideoCapture(vid)
        count = 0
        counter += 1
        success = True
        while success:
            success, image = cap.read()
            if count % 120 == 0:
                cv2.imwrite(pathOut + 'frame%d.jpg' % len(os.listdir(pathOut)), image)
            count += 1


"""
Extract faces from Frames directory
and save the faces in Faces directory
"""


def extract_faces():
    dir = "./Workspace/Frames/"
    outPath = "./Workspace/Faces/"
    listing = os.listdir(dir)
    for frame in listing:
        print("Scanning " + frame)
        file = dir + frame
        # Load image with cv2 to crop it
        face = cv2.imread(file)
        # Load image with Face_Recognition to scan for faces
        image = fr.load_image_file(file)
        # Save face locations
        face_locations = fr.face_locations(image)
        n = 0
        # Save each face found
        for loc in face_locations:
            x1 = loc[0]
            x2 = loc[2]
            y1 = loc[3]
            y2 = loc[1]
            # cv2.imshow("1", face[x1-50:x2+50,y1-50:y2+50])
            try:
                cv2.imwrite(outPath + "{}_{}.jpg".format(frame[:-4], n), face[x1 - 20:x2 + 20, y1 - 20:y2 + 20])
            except:
                print("")
            n += 1


"""
sort faces in Faces folder and move them into Faces_Sorted folder
"""


def sort_faces():
    dir = "./Workspace/Faces/"
    faces = os.listdir(dir)
    # save encoding of each face
    IDs = []
    encodings = []
    for i in range(len(faces)):
        # Add encoding
        img = cv2.imread(dir + faces[i])
        print(dir + faces[i])
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # catch face not found error
        try:
            img_encoding = fr.face_encodings(rgb_img)[0]
            encodings.append(img_encoding)
            IDs.append(0)
        except IndexError:
            # 128 to avoid dim error
            encodings.append(np.zeros(128))
            IDs.append(0)
    # Give each similar face the same ID
    frstID = 1
    print(IDs)
    for i in range(len(faces)):
        if IDs[i] == 0:
            IDs[i] = frstID
            results = fr.compare_faces(encodings, encodings[i])
            for j in range(len(results)):
                if results[j] == True and IDs[j] == 0:
                    IDs[j] = IDs[i]
            frstID += 1

    # Create folder for each unique ID
    uID = np.unique(IDs)
    for nFile in uID:
        newpath = "./Workspace/Faces_Sorted/{}".format(nFile)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    # move faces to proper folder
    for i in range(len(faces)):
        src = "./Workspace/Faces/{}".format(faces[i])
        # add len to filename so we dont delete files
        dst = "./Workspace/Faces_Sorted/{}/{}_{}".format(IDs[i],
                                                         len(os.listdir("./Workspace/Faces_Sorted/{}".format(IDs[i]))),
                                                         faces[i])
        shutil.move(src, dst)

"""
Find the given face name in the vidoes at the Vids_To_Scan folder
"""

def find_face(name):
    dir = "./Workspace/Vids_To_Scan/"
    print("./Workspace/Faces_Sorted/{}".format(name))
    # Check if the person is in the dataset
    file_exists = exists("./Workspace/Faces_Sorted/{}".format(name))
    if file_exists == False:
        print("Person isn't in the dataset")
        return
    # sfr.load_encoding_images("./Workspace/Faces_Sorted/{}".format(name))
    sfr.load_encoding_images("./Workspace/Faces_Sorted/")
    listing = os.listdir(dir)
    count = 0
    counter = 1
    for vid in listing:
        print("Scanning " + vid)
        vid = dir + vid
        cap = cv2.VideoCapture(vid)
        count = 0
        counter += 1
        success = True
        while success:
            success, image = cap.read()
            if count % 60 == 0:
                face_locations, face_names = sfr.detect_known_faces(image)
                if name in face_names:
                    print("{} found in {} at {}".format(name, vid.split("/")[-1], count // 60))
                    with open(dir + 'search result.txt', 'a') as f:
                        f.write("{} found in {} at {}:{}\n".format(name, vid.split("/")[-1], count % 60, count // 60))
            count += 1

# extract_frames()
# extract_faces()
# sort_faces()
# find_face()