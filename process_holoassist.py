import json
import multiprocessing
import os 

import cv2
import numpy as np

import csv

from tqdm import tqdm

def toString(text):
    if type(text) is list:
        text = ' '.join([str(elem) for elem in text])
    return text

def process_batch(ann):

    vid_name = ann["video_name"]       
    print(vid_name)

    #if (vid_name != "z176-sep-05-22-knarrevik_assemble"):
    #    continue
        
    vid_path = os.path.join(input_data_path, vid_name, "Export_py/Video.mp4")

    video_opened = False

    tries = 0
    while (video_opened == False):
        #print("Trying to open video file %s" % vid_path)  
        video = cv2.VideoCapture(vid_path)
        video_opened = video.isOpened()

        tries += 1

        if (video.isOpened()== False):
            print("Error opening video file %s, try again" % vid_path)  
        elif (tries > 3):
            print("Error opening video file %s, give up" % vid_path)  
            return []
        else:
            #print("Opened video file %s" % vid_path)  
            break

    fps = video.get(cv2.CAP_PROP_FPS)

    #print("fps %f" % (fps))

    events = ann["events"]

    data = []

    for event in events:
        #if (event["label"] == "Fine grained action" and event["id"] == "6c0041fe-a07d-4b16-82a5-65ef288ae2b0"):
        if (event["label"] == "Fine grained action"):
            attr = event["attributes"]

            if (attr["Action Correctness"] == "Correct Action"):
                #print(event)
                    
                if (useTimes):
                    startTime = event["startTime"]
                    endTime = event["endTime"]

                    startFrame = int(np.round(startTime*fps))
                    endFrame = int(np.round(endTime*fps))
                else:
                    startFrame = event["startTimeOriginalFPS"]
                    endFrame = event["endTimeOriginalFPS"]                    

                if ("Adjective" in attr):
                    prompt = toString(attr["Verb"]) + " " + toString(attr["Adjective"]) + " " + toString(attr["Noun"])
                else:
                    prompt = toString(attr["Verb"]) + " " + toString(attr["Noun"])

                prompt = prompt.lower().replace("_", " ")

                #print(attr)
                #print(prompt)

                if (playClip):

                    if (useTimes):
                        video.set(cv2.CAP_PROP_POS_MSEC, startTime*1000)
                        curr = startTime
                        end = endTime

                        print("%f\t%f\t%s" % (curr*fps, end*fps, prompt))
                    else:
                        video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
                        curr = startFrame
                        end = endFrame

                        print("%f\t%f\t%s" % (curr, end, prompt))

                    while (curr < end):
                        #path = os.path.join(vid_name, "Export_py/Video/%06d.png" % f)

                        #img = cv2.imread(os.path.join(input_data_path, path))
                        ret, img = video.read()
                            
                        if (useTimes):
                            curr = video.get(cv2.CAP_PROP_POS_MSEC)/1000
                        else:
                            curr = video.get(cv2.CAP_PROP_POS_FRAMES)

                        #print("%d" % curr)

                        img = cv2.putText(img, prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        cv2.imshow("Image", img)      
                        cv2.waitKey(100)      

                # get before and after
                else:

                    if (useTimes):
                        #shift back a second earlier
                        if (startFrameOneSecShift):
                            startTime = startTime-1
                            startFrame = int(np.round(startTime*fps))

                        video.set(cv2.CAP_PROP_POS_MSEC, startTime*1000)
                        ret1, img1 = video.read()

                        video.set(cv2.CAP_PROP_POS_MSEC, endTime*1000)
                        ret2, img2 = video.read() 
                    else:
                        #shift back a second earlier
                        if (startFrameOneSecShift):
                            startFrame = int(np.round(startFrame-fps))

                        video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
                        ret1, img1 = video.read()

                        video.set(cv2.CAP_PROP_POS_FRAMES, endFrame)
                        ret2, img2 = video.read()     

                    ## concatenate image Horizontally
                    #img_pair = np.concatenate((img1, img2), axis=1)                        
                    #img_pair = cv2.putText(img_pair, prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if (not ret1 or not ret2):
                        continue

                    if (saveData):
                        start_file = vid_name + "_%06d.jpg" % startFrame
                        end_file = vid_name + "_%06d.jpg" % endFrame

                        pair_file = vid_name + "_pair_%06d-%06d_%s.jpg" % (startFrame, endFrame, prompt)

                        start_path = os.path.join(output_dir, start_file)
                        end_path = os.path.join(output_dir, end_file)
                        
                        #print("%s\t%s\t%s" % (start_path, end_path, prompt))

                        cv2.imwrite(start_path, img1, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        cv2.imwrite(end_path, img2, [cv2.IMWRITE_JPEG_QUALITY, 80])

                        #pair_path = os.path.join(output_dir_pairs, pair_file)
                        #cv2.imwrite(pair_path, img_pair, [cv2.IMWRITE_JPEG_QUALITY, 80])

                        row = [start_file, end_file, prompt]

                        data.append(row)
    
                    if (showImages):
                        cv2.imshow("Image", img_pair)      
                        cv2.setWindowTitle("Image", prompt)
                        cv2.waitKey(3000)      

    return data

from tqdm.contrib.concurrent import process_map, thread_map

from multiprocessing import Pool
import time
from tqdm import *

if __name__ == "__main__":

    playClip = False
    useTimes = True
    startFrameOneSecShift = False
    saveData = True
    showImages = False
    multiprocess = True

    input_dir = "/mnt/e/Research/HoloAssist"
    input_data_path = "/mnt/hl2data"

    output_dir = "/mnt/e/Research/HoloAssist/holoassist_instruct-pix2pix/images"
    output_dir_pairs = "/mnt/e/Research/HoloAssist/holoassist_instruct-pix2pix/pairs"

    with open(os.path.join(input_dir, "labels_20230225_2221_fixed_typos.json"), "r") as f:
      labels = json.load(f)

    start = 0

    print("num batches %d" % len(labels))

    if (multiprocess):
        n_procs = 2

        #data = process_map(process_batch, labels, chunksize=1, max_workers=n_procs)

        data = []

        with Pool(processes=n_procs) as p:
                for i, results in tqdm(enumerate(p.imap(process_batch, labels), start), initial=start, total=len(labels)):
                    
                    data += results

                    with open(os.path.join(output_dir, 'metadata_%06d.csv' % i), 'w', newline='\n') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')

                        headers = ["image","image_target","text"]
                        writer.writerow(headers)
                        writer.writerows(data)

        #tqdm.set_lock(TRLock())
        #with ThreadPoolExecutor(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        #    data = p.map(partial(progresser, progress=True, write_safe=True, blocking=False), L)

    else:
        for i in tqdm(range(len(labels))):

            try:

                ann = labels[i]
                data = process_batch(ann)

                with open(os.path.join(output_dir, 'metadata_%06d.csv' % i), 'w', newline='\n') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')

                    headers = ["image","image_target","text"]
                    writer.writerow(headers)
                    writer.writerows(data)

            except:
                pass
