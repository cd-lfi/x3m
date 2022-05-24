import subprocess
import time

import cv2 as cv
import numpy as np

# conda install -c conda-forge ffmpeg
rtsp_url = "rtsp://localhost:8554/video"


width = 320
height = 320
fps = 20
command = ["ffmpeg",
           "-y",
           "-f", "rawvideo",
           "-vcodec", "rawvideo",
           "-pix_fmt", "bgr24",
           "-s", "{}x{}".format(width, height),
           "-r", str(fps),
           "-i", "-",
           "-c:v", "libx264",
           "-pix_fmt", "yuv420p",
           "-preset", "ultrafast",
           "-f", "rtsp",
           rtsp_url]


pipe = subprocess.Popen(command, stdin=subprocess.PIPE)


def todo():
    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv.putText(frame, f"{time.time():.3f}", (5, 35),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        pipe.stdin.write(frame.tobytes())

        key = cv.waitKey(30)
        if key == ord("q"):
            break

    return 0


if __name__ == "__main__":
    todo()
