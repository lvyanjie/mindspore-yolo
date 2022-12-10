from easydict import EasyDict as edit

config = edit()
config.input_size = 640    # 网络输入尺寸
config.input_channel = 3    # 
config.test_img_shape = 640

config.anchor_scales = [[12, 16],
                        [19, 36],
                        [40, 28],
                        [36, 75],
                        [76, 55],
                        [72, 146],
                        [142, 110],
                        [192, 243],
                        [459, 401]]
config.max_box = 150 # 数量
config.num_classes = 4
config.out_channels = (config.num_classes + 5) * 3  # 定值
config.jitter = 0.3
config.hue = 0.015
config.saturation = 1.5
config.value = 0.4   # 不解其意
config.label_smooth = 0
config.label_smooth_factor = 0.1
config.ignore_threshold = 0.7
config.momentum = 0.9
config.weight_decay = 0.0005
config.loss_scale = 1024

# 根据不同的version选取响应的input_shape
config.input_shape=[[3, 32, 64, 128, 256, 512, 1],
                    [3, 48, 96, 192, 384, 768, 2],
                    [3, 64, 128, 256, 512, 1024, 3],
                    [3, 80, 160, 320, 640, 1280, 4]]

# train
config.log_interval = 100
config.batch_size = 8   # train batchsize
config.lr_scheduler = "cosine_annealing"
config.max_epoch = 300
config.T_max = 300     #??
config.lr_epochs = "220,250"
config.image_path = "obs://4in1/4in1s/images/"   # 桶地址
config.anno_path = "obs://4in1/4in1s/labels/"
config.train_ids = "obs://4in1/4in1s/train.txt"
config.val_ids = "obs://4in1/4in1s/val.txt"
config.yolo_version = 'yolov5s'
config.output_dir = "obs://4in1/models/"


# test
config.test_ignore_threshold = 0.001
config.labels = ['person', 'calling', 'play_phone', 'smoke']
config.eval_nms_thresh = 0.3
