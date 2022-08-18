import requests
import json
import base64
import io
import argparse
import numpy as np
from threading import Thread
from collections import deque
import time
import cv2



def read_frame(args):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # If Reading a camera, we convert to int
    try:
        device  = int(args.input)
    except:
        device  = args.input

    cap = cv2.VideoCapture(device)
    
    # Check if video opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    frame_cnt = 0
    
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
    
        # Encode the frame into byte data
        data = cv2.imencode('.jpg', frame)[1].tobytes()
        queue.append(data)
        frame_cnt += 1

        # For videos, add a sleep so that we read at 30 FPS
        if not isinstance(device, int):
            time.sleep(1.0/30)
    
      # Break the loop
      else: 
        break

    print("Done reading {} frames".format(frame_cnt)) 

    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

def flush_frames(payload, snd_cnt):
    snd_cnt += len(payload)
    payload = json.dumps(payload)
    response = requests.post(api, data=payload, headers=headers)

    return snd_cnt

def send_frame(args):

  
  count = 0
  exit_cnt = 0
  payload = {}
  snd_cnt = 0
  start_time = time.time()
  fps = 0

  while True:

      if len(queue) == 0:
          exit_cnt += 1
          if exit_cnt >= 1000:
              print("Length of queue is {} , snd_cnt is {}".format(len(queue), snd_cnt))
              break

      # Batch the frames into a dict payload
      while queue and count < args.batch_size:
          data = queue.popleft()
          im_b64 = base64.b64encode(data).decode("utf8")
          payload[str(count)] = im_b64
          count += 1 

      if count >= args.batch_size:
          snd_cnt = flush_frames(payload, snd_cnt)

          # Calculate FPS
          end_time = time.time()
          fps = 1.0*args.batch_size/(end_time - start_time)
          if snd_cnt%20 == 0:
            print("With Batch Size {}, FPS at frame number {} is {:.1f}".format(args.batch_size, snd_cnt, fps))
            #print(response.content.decode("UTF-8"))

          # Reset for next batch
          start_time = time.time()
          count = 0 
          payload = {}
      
      # Sleep for 10 ms before trying to process next frame
      time.sleep(0.01)

  snd_cnt = flush_frames(payload, snd_cnt)
  print("With Batch Size {}, FPS at frame number {} is {:.1f}".format(args.batch_size, snd_cnt, fps))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="Batch frames on client side for inference",
                        type=int, default=1)
    parser.add_argument("--input", help="Path to video file or device id",
                        default='examples/image_classifier/streaming_video_client_side_batching/data/cat_vid.mp4')
    args = parser.parse_args()
    queue = deque([])
    api = 'http://localhost:8080/predictions/resnet-18'
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    thread1 = Thread(target=read_frame, args=(args,))
    thread2 = Thread(target=send_frame, args=(args,))
    thread1.start()
    thread2.start()