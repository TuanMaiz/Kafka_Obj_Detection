from datetime import datetime
from confluent_kafka import Producer
from serde import encodeToRaw
from dotenv import load_dotenv
import argparse
import cv2
import sys
import os

def delivery_callback(err, msg):
    if err:
        sys.stderr.write("%% Message failed delivery: %s\n" % err)
    else:
        sys.stderr.write(
            "%% Message delivered to %s [%d] @ %d\n"
            % (msg.topic(), msg.partition(), msg.offset())
        )

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
    vid = cv2.VideoCapture(args.video)

    try:
        while vid.isOpened():
            ret, frame = vid.read()
            assert ret       
            try:
                p.produce(
                    args.topic, 
                    encodeToRaw(frame, str(args.partition)),
                    callback=delivery_callback, 
                    partition=args.partition
                )
            except BufferError:
                print("some thing when wrong")
            p.poll(1)
    except:
        print("\nExiting.")
        vid.release()
        cv2.destroyAllWindows()
        p.flush()
        sys.exit(1)