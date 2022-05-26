import time
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np


def pull_rtsp(rtsp_url, frames=5, keep=False):
    cap = cv.VideoCapture(rtsp_url)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"INFO: {fps=}, {width=}, {height=}")

    if keep:
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        Path("tmp").mkdir(parents=True, exist_ok=True)
        out = cv.VideoWriter("tmp/rtsp_video.mp4",
                             fourcc, fps, (width, height))

    while True:
        if cap.isOpened():
            try:
                ret, frame = cap.read()
                if keep:
                    # cv.imwrite(f"tmp/IM{time.time():.3f}_rtsp.png", frame)
                    out.write(frame)
                else:
                    cv.imshow("Image", frame)
            except Exception as e:
                print(f"ERROR: {traceback.format_exc()}")
                cap = cv.VideoCapture(rtsp_url)
                time.sleep(1)
        else:
            print(f"N: can't open [{rtsp_url=}]")

        frames -= 1
        key = cv.waitKey(30)
        if frames < 1 or key == ord("q"):
            break

    cv.destroyAllWindows()
    if keep:
        out.release()
    cap.release()
    return 0


def test_imshow(frames=5, keep=False):
    fps = 30
    width = 320
    height = 320

    if keep:
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        Path("tmp").mkdir(parents=True, exist_ok=True)
        out = cv.VideoWriter("tmp/rtsp_video.mp4",
                             fourcc, fps, (width, height))

    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv.putText(frame, f"{time.time():.3f}", (5, 35),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        if keep:
            # cv.imwrite(f"tmp/IM{time.time():.3f}_test.png", frame)
            out.write(frame)
        else:
            cv.imshow("Image", frame)

        frames -= 1
        key = cv.waitKey(30)
        if frames < 1 or key == ord("q"):
            break

    cv.destroyAllWindows()
    if keep:
        out.release()
    return 0


if __name__ == "__main__":
    import sys
    task = sys.argv[1]
    rtsp_url = "rtsp://localhost:8554/video"

    if task == "rtsp":
        pull_rtsp(rtsp_url, frames=500, keep=True)
    else:
        test_imshow(frames=500, keep=True)
