from confluent_kafka import Consumer, KafkaException, TopicPartition
from serde import decodeFromRaw
from datetime import datetime
from dotenv import load_dotenv
import argparse
import cv2
import torch
import numpy as np
import time
import os

# Check if consumer is none or not
def filterNone(consumer):
    return False if consumer is None else True

# get message from each partition in topic
def poll(consumer):
    msg = consumer.poll(timeout=1.0)
    return msg
# consGroup = [c1, c2]

# Convert model type to np.array
def score_frame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
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
        return str(int(x))

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
                cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

if __name__ == "__main__":

    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("topic", type=str)
    args = parser.parse_args()

    load_dotenv()
    broker = os.environ.get("BROKER")
    group = "detection_group"
    topic = args.topic
    conf = {"bootstrap.servers": broker,
            "group.id": group,
            "auto.offset.reset": "earliest",}

    consGroup = []
    for partition in range(2):
        c = Consumer(conf)
        # Subscribe to topics
        c.assign([TopicPartition(topic, partition)])
        consGroup.append(c)

    # c = Consumer(conf)
    # c.subscribe([topic])

    try:
        while True:
            msgs = map(poll, consGroup)
            print(list(msgs), type(list(msgs)))
            for msg in msgs:
                if msg is None:
                    print("Waiting for message...")
                elif msg.error():
                    raise KafkaException(msg.error())
                else:
                    start_time = time.perf_counter()
                    # --
                    msg = decodeFromRaw(msg.value())
                    results = score_frame(msg["img"])
                    frame = plot_boxes(results, msg["img"])
                    # --
                    end_time = time.perf_counter()

                    id = msg["cameraID"]
                    timing = str(datetime.fromtimestamp(int(msg["timestamp"])))
                    fps = 1 / np.round(end_time - start_time, 3)
                    cv2.putText(frame, id + " " + f'FPS: {int(fps)}' + " " + timing, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (172, 7, 237), 2)
                    cv2.imshow("Public camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
    except KeyboardInterrupt:
        pass
    finally:
        # Close down consumer to commit final offsets.
        for cGr in consGroup:
            cGr.close()
        cv2.destroyAllWindows()