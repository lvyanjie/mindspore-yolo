#coding:utf-8
#训练脚本

import time
import os

import moxing as mox
import mindspore as ms
from mindspore import nn

from config import config as cfg
from utils import AverageMeter, get_param_groups
from yolo import YOLOV5, YoloWithLossCell
from initializer import default_recurisive_init
from dataset import create_yolo_dataset
from lr_scheduler import get_lr

lr_epochs = list(map(int, cfg.lr_epochs.split(',')))
image_dir = cfg.image_path
anno_dir = cfg.anno_path
train_ids_path = cfg.train_ids
val_ids_path = cfg.val_ids
model_save_train_machine = "ckpt_dirs"
os.makedirs(model_save_train_machine, exist_ok=True)

# set context
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend', device_id ='0')

loss_meter = AverageMeter('loss')
dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
network = YOLOV5(is_training=True, version=dict_version[cfg.yolo_version])
# 无任何pretrain权重
default_recurisive_init(network)
network = YoloWithLossCell(network)

train_ds = create_yolo_dataset(image_dir=cfg.image_path,
                        anno_dir=cfg.anno_path,
                        image_ids=cfg.train_ids,
                        batch_size=cfg.batch_size,
                        config=cfg,
                        is_training=True)
steps_per_epoch = train_ds.get_dataset_size()   # 数据集大小
lr = get_lr(cfg, steps_per_epoch)
opt = nn.Momentum(params=get_param_groups(network), momentum=cfg.momentum, learning_rate=ms.Tensor(lr), \
    weight_decay=cfg.weight_decay, loss_scale=cfg.loss_scale)   # 动量
network = nn.TrainOneStepCell(network, opt, cfg.loss_scale // 2)
network.set_train()

data_loader = train_ds.create_tuple_iterator(do_copy=False)
first_step = True
t_end = time.time()

for epoch_idx in range(cfg.max_epoch):
    for step_idx, data in enumerate(data_loader):
        images = data[0]
        input_shape = images.shape[2:4]
        input_shape = ms.Tensor(tuple(input_shape[::-1]), ms.float32)
        loss = network(images, data[2], data[3], data[4], data[5], data[6],
                        data[7], input_shape)
        loss_meter.update(loss.asnumpy())

        # it is used for loss, performance output per config.log_interval steps.
        if (epoch_idx * steps_per_epoch + step_idx) % cfg.log_interval == 0:
            time_used = time.time() - t_end
            if first_step:
                fps = cfg.batch_size / time_used
                per_step_time = time_used * 1000
                first_step = False
            else:
                fps = cfg.batch_size * cfg.log_interval / time_used
                per_step_time = time_used / cfg.log_interval * 1000
            print('epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, '
                                'lr:{}, per step time: {}ms'.format(epoch_idx + 1, step_idx + 1,
                                                                    loss_meter, fps, lr[step_idx], per_step_time))
            t_end = time.time()
            loss_meter.reset()
    
    ckpt_name = os.path.join(model_save_train_machine, "yolov5_{}_{}.ckpt".format(epoch_idx + 1, steps_per_epoch))
    ms.save_checkpoint(network, ckpt_name)
    # 将训练端文件拷贝到桶服务器中
    mox.file.copy(ckpt_name, cfg.output_dir)

print('==========end training===============')