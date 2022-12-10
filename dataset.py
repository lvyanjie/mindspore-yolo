#coding:utf-8
import cv2
import os
import random
import numpy as np
import multiprocessing
import moxing as mox
import mindspore.dataset as ds

from utils import xywhn2xyxy
from distributed_sampler import DistributedSampler
from transforms import MultiScaleTrans, PreprocessTrueBox, reshape_fn

# 已测试ok
class Dataset:
    """
    只执行训练过程，不进行验证
    :param image_dir: 图像存储路径, obs路径
    :param anno_dir: annotation存储路径  obs路径
    :param image_ids: 图像列表ids, txt格式
    :param is_training: 是否是训练状态
    """
    def __init__(self, image_dir, anno_dir, image_ids, input_size=640, is_training=True):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.image_ids = self._get_image_ids(image_ids)
        self.is_training = is_training
        self.mosaic = True
        self.input_size = input_size
        
        
    def _get_image_ids(self, image_ids):
        
        all_ids = []   # 用于存储所有的
        with mox.file.File(image_ids, 'r') as f:
            data = f.readlines()
        for img_id in data:
            img_id = img_id.replace(".jpg\r\n", "").split("/")[-1]
            all_ids.append(img_id)
        return all_ids
    
    def _get_target(self, anno_path):
        target = {}
        
        with mox.file.File(anno_path, 'r') as f:
            data = f.readlines()
        labels, bboxes = [], []
        for item in data:
            item = item.replace("\r\n", "")
            ss = item.split(" ")
            ss = [float(kk) for kk in ss]
            labels.append(ss[0])
            bboxes.append(ss[1:])
            
        target["bboxes"] = np.array(bboxes)
        target["labels"] = np.array(labels)
        return target
    
    # 加入
    def _mosaic_preprocess(self, index):
        """
        数据合成处理
        """
        labels4 = []
        s = self.input_size  # 这里有出入，和源码保持一致， 华为方提供的代码 s=384
        self.mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2*s+x)) for x in self.mosaic_border]   # 对应范围内产生的两个随机数
        indices = [index] + [random.randint(0, len(self.image_ids)-1) for _ in range(3)]   # 再获取三张图像数据
        for i, img_ids_index in enumerate(indices):
            img_id = self.image_ids[img_ids_index]
            img_path = os.path.join(self.image_dir, img_id+".jpg")
            anno_path = os.path.join(self.anno_dir, img_id+'.txt')
            img = cv2.imdecode(np.fromstring(mox.file.read(img_path, binary=True), np.uint8), cv2.IMREAD_COLOR)
            img = img[:,:,::-1]   # convert to RGB
            
            h, w = img.shape[:2]

            if i==0:   # top left
                img4 = np.full((s*2, s*2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h 
            if i==1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc+w, s*2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b

            target = self._get_target(anno_path)

            bboxes = target['bboxes']
            labels = target['labels']
            
            bboxes = xywhn2xyxy(bboxes, w=w, h=h, padw=padw, padh=padh)
            labels = np.reshape(labels, (len(labels), 1)).astype(np.int32)
            
            out_target = np.hstack([bboxes, labels])
            labels4.append(out_target)

        if labels4:
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, :4], 0, 2 * s, out=labels4[:, :4])  # use with random_perspective

        flag = np.array([1])
        # 为了和华为处理对应起来
        return img4, labels4, [self.input_size, self.input_size], flag
    
    
    def __getitem__(self, index):
        """
        iterator
        """
        img_id = self.image_ids[index]
        img_path = os.path.join(self.image_dir, img_id+".jpg")

        if not self.is_training:
            img = cv2.imdecode(np.fromstring(mox.file.read(img_path, binary=True), np.uint8), cv2.IMREAD_COLOR)
            img = img[:,:,::-1]
            return img, img_id

        anno_path = os.path.join(self.anno_dir, img_id+".txt")
        
        if self.mosaic and random.random() < 0.5:
            return self._mosaic_preprocess(index)

        img = cv2.imdecode(np.fromstring(mox.file.read(img_path, binary=True), np.uint8), cv2.IMREAD_COLOR)
        # convert img to RGB mode
        img = img[:,:,::-1]
        h, w = img.shape[:2]
        # 获取当前图像的所有boxes以及labels
        target = self._get_target(anno_path)
        bboxes, labels = target['bboxes'], target['labels']
        bboxes = xywhn2xyxy(bboxes, w=w, h=h, padw=0, padh=0)
        labels = np.reshape(labels, (len(labels), 1)).astype(np.int32)
        
        out_target = np.hstack([bboxes, labels])
        
        flag = np.array([0])
        return img, out_target, [self.input_size, self.input_size], flag
    
    
    def __len__(self):
        return len(self.image_ids)
            
        
# 无验证过程？
def create_yolo_dataset(image_dir, anno_dir, image_ids, batch_size, \
                       config=None, is_training=True, shuffle=True):
    """Create dataset"""
    cv2.setNumThreads(0)  # 关闭opencv的多线程功能
    # Enable shared memory feature to improve the performance of Python multiprocessing.
    ds.config.set_enable_shared_mem(True)
    device_num = 1   # 1个设备
    rank = 0   # only one device

    # 返回值：处理后图像, [bboxes,labels], [self.input_size, self.input_size], flag-mosaic
    yolo_dataset = Dataset(image_dir=image_dir, anno_dir=anno_dir, image_ids=image_ids, \
        input_size=config.input_size, is_training=is_training)

    # 采样器
    distributed_sampler = DistributedSampler(len(yolo_dataset), device_num, rank, shuffle=shuffle)
    # 通道转换
    hwc_to_chw = ds.vision.c_transforms.HWC2CHW()
    # 并行处理器个数
    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)
    if is_training:
        multi_scale_trans = MultiScaleTrans(config, device_num)
        dataset_column_names = ["image", "annotation", "input_size", "mosaic_flag"]
        output_column_names = ['image', 'annotation', 'bbox1', 'bbox2', 'bbox3', 'gt_box1', 'gt_box2', 'gt_box3']

        map1_out_column_names = ["image", "annotation", "size"]
        map1_column_order = ['image', 'annotation', 'size']
        map2_in_column_names = ['image', 'annotation', 'size']
        map2_out_column_names = ['image', 'annotation', 'bbox1', 'bbox2', 'bbox3', 'gt_box1', 'gt_box2', 'gt_box3']
        map2_column_order = ['image', 'annotation', 'bbox1', 'bbox2', 'bbox3', 'gt_box1', 'gt_box2', 'gt_box3']

        dataset = ds.GeneratorDataset(yolo_dataset, column_names=dataset_column_names, sampler=distributed_sampler,\
            num_parallel_workers=min(4, num_parallel_workers), python_multiprocessing=True)
        dataset = dataset.map(operations=multi_scale_trans, input_columns=dataset_column_names, \
            output_columns=map1_out_column_names, column_order=map1_column_order, \
                num_parallel_workers=min(12, num_parallel_workers), python_multiprocessing=True)
        dataset = dataset.map(operations=PreprocessTrueBox(config), input_columns=map2_in_column_names, \
            output_columns=map2_out_column_names, column_order=map2_column_order,\
                num_parallel_workers=min(4, num_parallel_workers), python_multiprocessing=False)
        dataset = dataset.project(output_column_names)

        # Computed from random subset of ImageNet training images
        mean = [m * 255 for m in [0.485, 0.456, 0.406]]
        std = [s * 255 for s in [0.229, 0.224, 0.225]]
        dataset = dataset.map([ds.vision.c_transforms.Normalize(mean, std), hwc_to_chw],\
            num_parallel_workers=min(4, num_parallel_workers))    # input_columns 为None，operation施加给第一列

        # 该处理将input image变成 [12, 320,320]，对应的gtboxes也不对了，为什么要这样做？
        # def concatenate(images):
        #     images = np.concatenate((images[..., ::2, ::2], images[..., 1::2, ::2],
        #                              images[..., ::2, 1::2], images[..., 1::2, 1::2]), axis=0)
        #     return images
        # dataset = dataset.map(operations=concatenate, input_columns="image",
        #                       num_parallel_workers=min(4, num_parallel_workers))
        dataset = dataset.batch(batch_size, num_parallel_workers=min(4, num_parallel_workers), drop_remainder=True)
    else:
        # 对方直接用的resize模式对图像数据进行了缩放，和训练数据不同步
        dataset = ds.GeneratorDataset(yolo_dataset, column_names=['image', 'img_id'],\
            sampler=distributed_sampler)
        compose_map_func = (lambda image, img_id: reshape_fn(image, img_id, config))
        dataset = dataset.map(operations=compose_map_func, input_columns=["image", "img_id"],
                              output_columns=["image", "image_shape", "img_id"],
                              column_order=["image", "image_shape", "img_id"],
                              num_parallel_workers=8)
        dataset = dataset.map(operations=hwc_to_chw, num_parallel_workers=8)
        dataset = dataset.batch(batch_size, drop_remainder=True)                      

    return dataset
