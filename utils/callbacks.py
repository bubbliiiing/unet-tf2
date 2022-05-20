import os
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir, val_loss_flag=True):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(logs.get('loss'))
        if self.val_loss_flag:
            self.val_loss.append(logs.get('val_loss'))
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(logs.get('val_loss')))
                f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate         = decay_rate
        self.verbose            = verbose
        self.learning_rates     = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))

class EvalCallback(keras.callbacks.Callback):
    def __init__(self, model_body, input_shape, num_classes, image_ids, dataset_path, log_dir,\
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.model_body         = model_body
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    @tf.function
    def get_pred(self, image_data):
        pr = self.model_body(image_data, training=False)
        return pr
    
    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别 
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   归一化+通道数调整到第一维度+添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        pr = self.get_pred(image_data)[0].numpy()
        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #--------------------------------------#
        #   进行图片的resize
        #--------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        if temp_epoch % self.period == 0 and self.eval_flag:
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(temp_epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
