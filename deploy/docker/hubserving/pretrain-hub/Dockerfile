FROM paddlepaddle/paddle:2.3.0

RUN pip install paddlehub

RUN hub install chinese_ocr_db_crnn_mobile chinese_text_detection_db_mobile

EXPOSE 8866

CMD ["/bin/bash", "-c", "hub serving start -m chinese_ocr_db_crnn_mobile -p 8866"]