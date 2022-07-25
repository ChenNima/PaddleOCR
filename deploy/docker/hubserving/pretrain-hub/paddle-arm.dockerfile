FROM python:3.8

COPY paddle/paddlepaddle-2.3.1-cp38-cp38-linux_aarch64.whl paddlepaddle-2.3.1-cp38-cp38-linux_aarch64.whl

RUN pip install -U paddlepaddle-2.3.1-cp38-cp38-linux_aarch64.whl

RUN rm paddlepaddle-2.3.1-cp38-cp38-linux_aarch64.whl

RUN apt update

RUN apt install cmake protobuf-compiler ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python==4.6.0.66 shapely pyclipper

ENTRYPOINT [ "/bin/bash" ]