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

def process_video(vid_name_frames):

    #print(vid_name)

    vid_name, frames = vid_name_frames
    
    frames = sorted(frames)

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

    for time, frame in frames:
                    
        if (useTimes):
            video.set(cv2.CAP_PROP_POS_MSEC, time*1000)
            ret1, img = video.read()
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret1, img = video.read()

        if (not ret1):
            continue

        file = vid_name + "_%06d.jpg" % frame
        path = os.path.join(output_dir, file)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])

def process_batch(ann):

    vid_name = ann["video_name"]       
    #print(vid_name)

    #if (vid_name != "z176-sep-05-22-knarrevik_assemble"):
    #    continue
    fps = ann["videoMetadata"]["video"]["fps"]

    events = ann["events"]

    data = []

    for event in events:
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
                    
                    startTime = startFrame/fps
                    endTime = endFrame/fps

                if ("Adjective" in attr):
                    prompt = toString(attr["Verb"]) + " " + toString(attr["Adjective"]) + " " + toString(attr["Noun"])
                else:
                    prompt = toString(attr["Verb"]) + " " + toString(attr["Noun"])

                prompt = prompt.lower().replace("_", " ")

                start_file = vid_name + "_%06d.jpg" % startFrame
                end_file = vid_name + "_%06d.jpg" % endFrame

                row = [start_file, end_file, prompt, vid_name, startTime, endTime, startFrame, endFrame]

                data.append(row)   

    data =sorted(data)

    return data

from tqdm.contrib.concurrent import process_map, thread_map

from multiprocessing import Pool
import time
from tqdm import *

from sys import platform

if __name__ == "__main__":

    useTimes = True
    multiprocess = True

    #playClip = False
    #startFrameOneSecShift = False
    #saveData = True
    #showImages = False

    if platform == "win32":
        input_dir = "E:/Research/HoloAssist"
        output_dir = "E:/Research/HoloAssist/holoassist_instruct-pix2pix/images"
    else:
        input_dir = "/mnt/e/Research/HoloAssist"
        output_dir = "/mnt/e/Research/HoloAssist/holoassist_instruct-pix2pix/images"

    input_data_path = "/mnt/hl2data"

    output_dir_pairs = "/mnt/e/Research/HoloAssist/holoassist_instruct-pix2pix/pairs"

    with open(os.path.join(input_dir, "labels_20230225_2221_fixed_typos.json"), "r") as f:
      labels = json.load(f)

    print("num batches %d" % len(labels))

    data = []

    for ann in tqdm(labels):
        try:
            data += process_batch(ann)
        except Exception as e:
            print(e)
            pass

    #print(len(data))
    #print(len(labels))

    with open(os.path.join(output_dir, 'metadata.csv'), 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        headers = ["image","image_target","text"]
        writer.writerow(headers)

        for row in data:
            writer.writerow(row[0:3])

    video_frames = {}

    for _, _, _, vid_name, startTime, endTime, startFrame, endFrame in data:

        if (vid_name not in video_frames):
            video_frames[vid_name] = set()

        video_frames[vid_name].add((startTime, startFrame))
        video_frames[vid_name].add((endTime, endFrame))

    #print(len(video_frames))

    if (multiprocess):
        n_procs = 2

        process_map(process_video, video_frames.items(), chunksize=1, max_workers=n_procs)

    else:
        for vid_name_frames in tqdm(video_frames.items()):
            process_video(vid_name_frames)
