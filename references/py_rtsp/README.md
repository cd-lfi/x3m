# RTSP
`conda install -c conda-forge ffmpeg` and launch python with `Anaconda Powershell Prompt (anaconda3)`.

## rtsp-server
https://github.com/topics/rtsp-server

[rtsp-simple-server](https://github.com/aler9/rtsp-simple-server):

1. launch rtsp server: `rtsp-simple-server.exe`
2. push: `ffmpeg -re -stream_loop -1 -i test.mp4 -c copy -f rtsp rtsp://localhost:8554/mystream`
3. pull: `ffmpeg -i rtsp://localhost:8554/mystream -c copy output.mp4`

## python demo
1. rtsp server: `rtsp-simple-server.exe`
2. `references/py_rtsp/push.py`
3. `references/py_rtsp/pull.py`
