import base64
import sys
import numpy as np
import paddle
import yaml
import json

sys.path.insert(0, ".")


import paddlehub as hub

from ppocr.data import create_operators, transform
from ppocr.utils.save_load import load_model
from ppocr.modeling.architectures import apply_to_static, build_model
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from tools.infer.utility import base64_to_cv2
import paddle.nn.functional as F
import paddle.fluid as fluid

def read_class_list(filepath):
    dict = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split(" ")
            dict[key] = value.rstrip()
    return dict

@moduleinfo(
    name="kie_system",
    version="1.0.0",
    summary="kie system service",
    author="felix.chen",
    author_email="felix.chen@sap.com",
    type="cv/PP-KIE_system")
class KIESystem(hub.Module):
    def _initialize(self, ocr_module=None, enable_mkldnn=False):
        fluid.enable_dygraph() 
        paddle.set_device("cpu")
        with open("./configs/kie/kie_unet_sdmgr_fapiao_hub.yml", "r") as stream:
            config = yaml.safe_load(stream)
            # config['Global']['use_gpu'] = False
            # config['Global']['checkpoints'] = "params/kie/best_accuracy"
            self.config = config
        global_config = config['Global']
        model = build_model(config['Architecture'])
        model = apply_to_static(model, config, logger)
        load_model(config, model)
        model.eval()
        self.model = model
        class_path = config['Global']['class_path']
        self.idx_to_cls = read_class_list(class_path)
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            transforms.append(op)
        self.ops = create_operators(transforms, global_config)
        self._ocr_module = ocr_module
        self.enable_mkldnn = enable_mkldnn

    @property
    def ocr_module(self):
        """
        ocr module
        """
        if not self._ocr_module:
            self._ocr_module = hub.Module(name='chinese_ocr_db_crnn_mobile', enable_mkldnn=self.enable_mkldnn)
        return self._ocr_module

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        fluid.enable_dygraph() 
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.ocr_module.recognize_text(images_decode, **kwargs)
        results = [{
            "image": base64.b64decode(images[index]),
            "label": [{
                    "label": 1,
                    "transcription": data["text"],
                    "points": data["text_box_position"]
                } for data in res["data"]],
        } for index, res in enumerate(results)]

        output = []
        for res in results:
            label = json.dumps(res["label"], ensure_ascii=False)
            data = {
                "label": label,
                "image": res["image"]
            }
            batch = transform(data, self.ops)
            batch_pred = [0] * len(batch)
            for i in range(len(batch)):
                batch_pred[i] = paddle.to_tensor(
                    np.expand_dims(
                        batch[i], axis=0))
            node, edge = self.model(batch_pred)
            node = F.softmax(node, -1)
            max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
            node_pred_label = max_idx.numpy().tolist()
            node_pred_score = max_value.numpy().tolist()
            annotations = res["label"]
            pred_res = {}
            for i, label in enumerate(node_pred_label):
                pred_score = '{:.2f}'.format(node_pred_score[i])
                pred_label = str(node_pred_label[i])
                if pred_label in self.idx_to_cls:
                    pred_label = self.idx_to_cls[pred_label]
                pred_item = {
                    'label': label,
                    'transcription': annotations[i]['transcription'],
                    'score': pred_score,
                    'points': annotations[i]['points'],
                }
                if pred_label in pred_res:
                    pred_res[pred_label].append(pred_item)
                else:
                    pred_res[pred_label] = [pred_item]
            output.append(pred_res)
        return output