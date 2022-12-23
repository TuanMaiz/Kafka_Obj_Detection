from confluent_kafka import Consumer, KafkaException, TopicPartition
from serde import decodeFromRaw
from dotenv import load_dotenv
import argparse
import cv2
import os

def consume(consumer):
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            print(f"Waiting for message {args.partition} ...")
        elif msg.error():
            raise KafkaException(msg.error())
        else:
            msg = decodeFromRaw(msg.value())
            cv2.imshow(f"camera {args.partition}", msg["img"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    
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

    consume(c)
