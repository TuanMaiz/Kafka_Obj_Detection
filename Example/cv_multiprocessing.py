from datetime import datetime
import concurrent.futures
import cv2
import torch
import numpy as np
import time

# RTSP_URL = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
# RTSP_List = [(RTSP_URL,0), (RTSP_URL,1), (RTSP_URL,2), (RTSP_URL,3)]

Video_List = [("Traffic1.m4v",0), ("Traffic2.m4v",1), ("Traffic3.m4v",2)]

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
            x1, y1, x2, y2 = int(
                row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, str(model.names[class_to_label(labels[i])]) + " " + str(round(float(row[4]),2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
    return frame

def url_to_video(tup):
    url,index = tup
    video = cv2.VideoCapture(url)
    try:
        while video.isOpened():
            start_time = time.perf_counter()
            ret, frm = video.read()
            if not ret: 
                break

            gray = cv2.resize(frm, (416, 416))
            results = score_frame(gray)
            frame = plot_boxes(results, gray)
            end_time = time.perf_counter()

            dt = str(datetime.now())
            fps = 1 / np.round(end_time - start_time, 3)

            cv2.putText(frame, f'FPS: {int(fps)}' + " "*2 + dt, (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.6, (172, 7, 237), 2)

            cv2.imshow(f"Video {index}", frame)
            cv2.moveWindow(f"Video {index}", index*500, 0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    while True:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(url_to_video, Video_List)
        cv2.destroyAllWindows()