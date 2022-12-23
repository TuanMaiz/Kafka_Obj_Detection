from datetime import datetime
from confluent_kafka import Producer
from serde import encodeToRaw
from dotenv import load_dotenv
import torch
import argparse
import cv2
import sys
import os
import numpy as np
import time

########################### Convert model to np.array
def score_frame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        model.to('cuda')
        #  if torch.cuda.is_available() else 'cpu'
        frame = [frame]
        results = model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

def class_to_label(x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return int(x)

def plot_boxes(results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, str(model.names[class_to_label(labels[i])]) + " " + str(round(float(row[4]),2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame
###############################################################

def delivery_callback(err, msg):
    if err:
        sys.stderr.write("%% Message failed delivery: %s\n" % err)
    else:
        sys.stderr.write(
            "%% Message delivered to %s [%d] @ %d\n"
            % (msg.topic(), msg.partition(), msg.offset())
        )

def send(id, url):
    vid = cv2.VideoCapture(url)
    try:
        while vid.isOpened():
            start_time = time.perf_counter()
            # --
            success, frm = vid.read()
            if not success:
                break
            gray = cv2.resize(frm, (416, 416))
            results = score_frame(gray)
            frame = plot_boxes(results, gray)
            # --
            end_time = time.perf_counter()

            dt = str(datetime.now())
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}' + " " + dt, (20, 70),cv2.FONT_HERSHEY_DUPLEX, 0.7, (172, 7, 237), 2)

            # encode msg and send to kafka topic
            p.produce(
                args.topic, 
                encodeToRaw(frame, str(id)),
                callback=delivery_callback, 
                partition=id
            )
            p.poll(1)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\npartition {id} is" + " Exiting.")
        vid.release()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", type=str)
    parser.add_argument("partition", type=int)
    parser.add_argument("video", type=str)
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    load_dotenv()
    broker = os.environ.get("BROKER")
    conf = {"bootstrap.servers": broker}
    p = Producer(**conf)

    send(args.partition, args.video)