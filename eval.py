# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YoloV5 eval."""
import os
import time

import numpy as np

import mindspore as ms

from yolo import YOLOV5
from utils import DetectionEngine
from dataset import create_yolo_dataset

from config import config as cfg


def statistic_normalize_img(img):
    """Statistic normalize images."""
    # img: RGB
    img = img / 255.
    # Computed from random subset of ImageNet training images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img


image_path = cfg.image_path
anno_path = cfg.anno_path
val_ids = cfg.train_ids   # 暂时先以train_ids进行测试，验证模型收敛情况
model_path = './ckpt_dirs/yolov5_73_88.ckpt'
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend', device_id ='0')

start_time = time.time()
print('Creating Network....')
dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
network = YOLOV5(is_training=False, version=dict_version[cfg.yolov5_version])

# load weights
param_dict = ms.load_checkpoint(model_path)
param_dict_new = {}
for key, values in param_dict.items():
    if key.startswith('moments.'):
        continue
    elif key.startswith('yolo_network.'):
        param_dict_new[key[13:]] = values
    else:
        param_dict_new[key] = values
ms.load_param_into_net(network, param_dict_new)
print("load_model %s sucess", model_path)

val_ds = create_yolo_dataset(image_dir=image_path,
                            anno_dir=anno_path,
                            image_ids=val_ids,
                            batch_size=1,
                            config=cfg,
                            is_training=False,
                            shuffle=False)

print('testing shape : %s', cfg.test_img_shape)
print('total %d images to eval', val_ds.get_dataset_size())

network.set_train(False)

# init detection engine
detection = DetectionEngine(cfg, cfg.test_ignore_threshold)
input_shape = ms.Tensor(tuple(cfg.test_img_shape), ms.float32)
print('Start inference....')
for index, data in enumerate(val_ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
    # 将数据转换为字典格式？
    image = data["image"]   # 只进行了reshape
    # adapt network shape of input data
    # 对image进行规范化处理
    image_norm = statistic_normalize_img(image)
    input_image = ms.Tensor(image_norm)
    image_shape_ = data["image_shape"]
    image_id_ = data["img_id"]
    output_big, output_me, output_small = network(input_image, input_shape)
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()
    # batch_size = 1
    detection.detect([output_small, output_me, output_big], batch=1, image_shape=image_shape_, image_id=image_id_)

    if index % 10 == 0:
        print('Processing... {:.2f}% '.format(index / val_ds.get_dataset_size() * 100))

    print('Calculating mAP...')
    detection.do_nms_for_results()
    # result_file_path = detection.write_result()
    # config.logger.info('result file path: %s', result_file_path)
    eval_result = detection.get_eval_result()

    cost_time = time.time() - start_time
    eval_log_string = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_log_string)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)


if __name__ == "__main__":
    run_eval()
