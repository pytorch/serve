import argparse
import time
from collections import deque
from concurrent.futures import as_completed
from threading import Thread

import cv2
from requests_futures.sessions import FuturesSession


def read_frames(args):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # If Reading a camera, we convert to int
    try:
        device = int(args.input)
    except:
        device = args.input

    cap = cv2.VideoCapture(device)

    # Check if video opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_cnt = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Encode the frame into byte data
            data = cv2.imencode(".jpg", frame)[1].tobytes()
            queue.append(data)
            frame_cnt += 1

            # For videos, add a sleep so that we read at 30 FPS
            if not isinstance(device, int):
                time.sleep(1.0 / 30)

        # Break the loop
        else:
            break

    print("Done reading {} frames".format(frame_cnt))

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def send_frames(payload, snd_cnt, session):
    snd_cnt += 1

    response = session.post(api, data=payload)

    return (response, snd_cnt)


def batch_and_send_frames(args):

    # Initialize variables
    count = 0
    exit_cnt = 0
    payload = {}
    snd_cnt = 0
    start_time = time.time()
    fps = 0
    session = FuturesSession()
    futures = []
    log_cnt = 20

    while True:

        # Exit condition for the while loop. Need a better logic
        if len(queue) == 0:
            exit_cnt += 1
            # By trial and error, 1000 seemed to work best
            if exit_cnt >= 1000:
                print(
                    "Length of queue is {} , snd_cnt is {}".format(len(queue), snd_cnt)
                )
                break

        # Batch the frames into a dict payload
        while queue and count < args.batch_size:
            data = queue.popleft()
            payload = data
            count += 1

        if count >= args.batch_size:
            response, snd_cnt = send_frames(payload, snd_cnt, session)
            futures.append(response)

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 * args.batch_size / (end_time - start_time)
            if snd_cnt % log_cnt == 0:
                print(
                    "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
                        args.batch_size, snd_cnt, fps
                    )
                )

                # Printing the response
                for response in list(as_completed(futures))[-4:]:
                    print(response.result().content.decode("utf-8"))

                # Cleaning up futures in case futures becomes too large
                del futures[:log_cnt]

            # Reset for next batch
            start_time = time.time()
            count = 0
            payload = {}

        # Sleep for 10 ms before trying to send next batch of frames
        time.sleep(args.sleep)

    # Send any remaining frames
    _, snd_cnt = send_frames(payload, snd_cnt, session)
    print(
        "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
            args.batch_size, snd_cnt, fps
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        help="Batch frames on TorchServe side for inference",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--input",
        help="Path to video file or device id",
        default="examples/image_classifier/streaming_video/data/sample_video.mp4",
    )
    parser.add_argument(
        "--sleep",
        help="Sleep between 2 subsequent requests in seconds",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()

    # Read frames are placed here and then processed
    queue = deque([])
    api = "http://localhost:8080/predictions/resnet-18"

    thread1 = Thread(target=read_frames, args=(args,))
    thread2 = Thread(target=batch_and_send_frames, args=(args,))
    thread1.start()
    thread2.start()
