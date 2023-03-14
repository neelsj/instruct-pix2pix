import json
import os 

import cv2
import numpy as np

import csv

from tqdm import tqdm

if __name__ == "__main__":

    input_dir = "/mnt/e/Research/HoloAssist/holoassist_samples"
    input_data_path = "/mnt/hl2data"

    output_dir = "/mnt/e/Research/HoloAssist/holoassist_instruct-pix2pix"

    with open(os.path.join(input_dir, "data_nov-03_psi.json"), "r") as f:
      ann = json.load(f)

    data = []

    for vid_name in tqdm(ann.keys()):
       
        #if (vid_name != "z047-june-25-22-nespresso"):
        #    continue

        events = ann[vid_name]["events"]
           
        video = cv2.VideoCapture(os.path.join(input_data_path, vid_name, "Export_py/Video.mp4"))

        if (video.isOpened()== False):
            print("Error opening video file")  
            continue

        fps = video.get(cv2.CAP_PROP_FPS)

        #print("fps %f" % (fps))

        useTimes = True
        shift = 2

        for event in events:
            #if (event["label"] == "Fine grained action" and event["id"] == "6c0041fe-a07d-4b16-82a5-65ef288ae2b0"):
            if (event["label"] == "Fine grained action"):
                attr = event["attributes"]

                if (attr["Action correctness"] == "Correct Action"):
                    #print(event)

                    #startFrame = event["start"]
                    #endFrame = event["end"]

                    #startTime = event["startTime"]*1000-2000
                    #endTime = event["endTime"]*1000-2000

                    ##start = int(np.round(startTime*fps,0))
                    ##end = int(np.round(endTime*fps,0))

                    #if ("Adjective" in attr):
                    #    prompt = attr["Verb"] + " " + attr["Adjective"] + " " + attr["Noun"]
                    #else:
                    #    prompt = attr["Verb"] + " " + attr["Noun"]

                    #if (useTimes):
                    #    video.set(cv2.CAP_PROP_POS_MSEC, startTime)
                    #    curr = startTime
                    #    end = endTime
                    #    print("%f\t%f\t%s" % (curr/1000, end/1000, prompt))
                    #else:
                    #    video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
                    #    curr = startFrame
                    #    end = endFrame
                    #    print("%d\t%d\t%s" % (curr, end, prompt))

                    #while (curr < end):
                    #    #path = os.path.join(vid_name, "Export_py/Video/%06d.png" % f)

                    #    #img = cv2.imread(os.path.join(input_data_path, path))
                    #    ret, img = video.read()
                        
                    #    if (useTimes):
                    #        curr = video.get(cv2.CAP_PROP_POS_MSEC)
                    #        #print("%f" % (curr/1000))
                    #    else:
                    #        curr = video.get(cv2.CAP_PROP_POS_FRAMES)
                    #        #print("%d" % curr)

                    #    cv2.imshow("Image", img)      
                    #    cv2.setWindowTitle("Image", prompt)
                    #    cv2.waitKey(100)      

                    if ("Adjective" in attr):
                        prompt = attr["Verb"] + " " + attr["Adjective"] + " " + attr["Noun"]
                    else:
                        prompt = attr["Verb"] + " " + attr["Noun"]

                    prompt = prompt.lower()

                    if (useTimes):

                        startTime = (event["startTime"]-shift)
                        endTime = (event["endTime"]-shift)
                        startTime = startTime-1

                        startFrame = int(np.round(startTime*fps))
                        endFrame = int(np.round( endTime*fps))

                        video.set(cv2.CAP_PROP_POS_MSEC, startTime*1000)
                        ret, img1 = video.read()

                        video.set(cv2.CAP_PROP_POS_MSEC, endTime*1000)
                        ret, img2 = video.read()
                    else:

                        startFrame = int(np.round(event["start"]-shift*fps))
                        endFrame = int(np.round(event["end"]-shift*fps))
                        startFrame = int(np.round(startFrame-fps))

                        video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
                        ret, img1 = video.read()

                        video.set(cv2.CAP_PROP_POS_FRAMES, endFrame)
                        ret, img2 = video.read()     

                    start_file = vid_name + "_%06d.jpg" % startFrame
                    end_file = vid_name + "_%06d.jpg" % endFrame

                    start_path = os.path.join(output_dir, start_file)
                    end_path = os.path.join(output_dir, end_file)

                    #print("%s\t%s\t%s" % (start_path, end_path, prompt))

                    cv2.imwrite(start_path, img1, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(end_path, img2, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    row = [start_file, end_file, prompt]

                    data.append(row)
    
                    ## concatenate image Horizontally
                    #Hori = np.concatenate((img1, img2), axis=1)
  
                    #cv2.imshow("Image", Hori)      
                    #cv2.setWindowTitle("Image", prompt)
                    #cv2.waitKey(3000)      

        #if (vid_name != "z047-june-25-22-nespresso"):
        #    break

    with open(os.path.join(output_dir, 'metadata2.csv'), 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        headers = ["image","image_target","text"]
        writer.writerow(headers)
        writer.writerows(data)
