from datetime import datetime
import cv2
import torch
import numpy as np
import time

def score_frame(frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    frame = [frame]
    results = model(frame)
    results.print()

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
            cv2.putText(frame, str(model.names[class_to_label(labels[i])]) + " " + str(round(float(row[4]),2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame

if __name__ == "__main__":

    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # fc = 0
    # FPS = 0
    # display_time = 2
    # start_time = time.time()

    vid = cv2.VideoCapture('Traffic1.m4v')
    # while vid.isOpened():
    #     ret, frm = vid.read()
    #     if not ret: 
    #         break
    #     modelframe = model(frm)
    #     results = score_frame(frm)
    #     frame = plot_boxes(results, frm)
    #     print("---------")
    #     print(type(modelframe))
    #     print("---------")
    #     print(type(frm))
    #     print("---------")
    #     print(type(frame))

    while vid.isOpened():
        """
        ret, frame = vid.read()
        assert ret
        results = score_frame(frame)
        frame = plot_boxes(results, frame)
        if not (frame is None):
            fc += 1
            TIME = time.time() - start_time
            if (TIME) >= display_time:
                FPS = fc / (TIME)
                fc = 0
                start_time = time.time()
            fps_disp = "FPS: " + str(FPS)[:5]
            image = cv2.putText(
                frame,
                fps_disp,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("frame", image)
        if cv2.waitKey(1) == ord("q"):
            break
        """
        start_time = time.perf_counter()
        # --
        ret, frm = vid.read()
        if not ret: 
            break
        results = score_frame(frm)
        frame = plot_boxes(results, frm)
        # --
        end_time = time.perf_counter()
        dt = str(datetime.now())
        fps = 1 / np.round(end_time - start_time, 3)
        cv2.putText(frame, f'FPS: {int(fps)}' + " "*14 + dt, (20, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (172, 7, 237), 2)
        cv2.imshow("Public camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    print("\nExiting.")
