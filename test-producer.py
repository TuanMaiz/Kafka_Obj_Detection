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
            success, frm = vid.read()
            if not success:
                break
            gray = cv2.resize(frm, (416, 416))        
            p.produce(
                args.topic, 
                encodeToRaw(gray, str(id)),
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

    load_dotenv()
    broker = os.environ.get("BROKER")
    conf = {"bootstrap.servers": broker}
    p = Producer(**conf)

    send(args.partition, args.video)