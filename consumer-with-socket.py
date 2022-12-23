from confluent_kafka import Consumer, KafkaException, TopicPartition
from serde import decodeFromRaw
from dotenv import load_dotenv
import argparse
import cv2
import os
import base64
import socketio
#socket io part
sio = socketio.Client()
@sio.event
def connect():
    print('connected to websocket server')
    print('my sid is', sio.get_sid())
@sio.event
def disconnect():
    print('disconnected from server')

def consume(consumer):
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            print(f"Waiting for message {args.partition} ...")
        elif msg.error():
            raise KafkaException(msg.error())
        else:
            msg = decodeFromRaw(msg.value())
            #encode to base64 then send to websocket server
            frame_b64 = base64.b64encode(msg["frame"]).decode('utf-8')
            sio.emit('send-frame', {'frame': frame_b64}) #send to websocket
            cv2.imshow(f"camera {args.partition}", msg["img"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def send_to_websocket(c):
    sio.connect('http://localhost:3001')
    consume(c)
    sio.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", type=str)
    parser.add_argument("partition", type=int)
    args = parser.parse_args()

    load_dotenv()
    broker = os.environ.get("BROKER")
    group = "detection_group"
    conf = {"bootstrap.servers": broker,
            "group.id": group,
            "auto.offset.reset": "earliest",}

    c = Consumer(conf)
    c.assign([TopicPartition(args.topic, args.partition)])
    send_to_websocket(c)