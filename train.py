import os
from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nets.unet import Unet
from nets.unet_training import CE, Generator, LossHistory, dice_loss_with_CE
from utils.metrics import Iou_score, f_score
from utils.utils import ModelCheckpoint

'''
需要注意的是，只要是训练自己的数据集，无论是自己的医药数据集还是普通数据集，都应该使用这个train.py进行训练。
格式需要符合VOC数据集的格式。

如果大家有自己的医药数据集需要标注后训练，完全可以按照视频里面的教程
首先利用labelme标注图片，转换称VOC格式后利用train.py进行训练。
训练自己标注的医药数据集的步骤和正常数据集的步骤一摸一样，不需要用到这个train_medical.py！
'''

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(images, labels, net, optimizer, loss):
        with tf.GradientTape() as tape:
            # 计算loss
            prediction = net(images, training=True)
            loss_value = loss(labels, prediction)

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        _f_score = f_score()(labels, prediction)
        return loss_value, _f_score
    return train_step

@tf.function
def val_step(images, labels, net, optimizer, loss):
    # 计算loss
    prediction = net(images, training=False)
    loss_value = loss(labels, prediction)
    
    _f_score = f_score()(labels, prediction)
    return loss_value, _f_score

def fit_one_epoch(net, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, train_step):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, labels = batch[0], batch[1]
            labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

            loss_value, _f_score = train_step(images, labels, net, optimizer, loss)
            total_loss          += loss_value.numpy()
            total_f_score       += _f_score.numpy()

            pbar.set_postfix(**{'Total Loss'        : total_loss / (iteration + 1), 
                                'Total f_score'     : total_f_score / (iteration + 1),
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
        
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            images, labels = batch[0], batch[1]
            labels = tf.convert_to_tensor(labels)

            loss_value, _f_score = val_step(images, labels, net, optimizer, loss)
            val_loss            += loss_value.numpy()
            val_f_score         += _f_score.numpy()

            pbar.set_postfix(**{'Val Loss'      : val_loss / (iteration + 1), 
                                'Val f_score'   : val_f_score / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss/(epoch_size+1), 'val_loss': val_loss/(epoch_size_val+1)}
    loss_history.on_epoch_end([], logs)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
if __name__ == "__main__":
    log_dir = "logs/"
    #----------------------------------------------------#
    #   是否使用eager模式训练
    #----------------------------------------------------#
    eager = True
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    inputs_size = [512,512,3]
    #------------------------------#
    #   分类个数+1
    #   2+1
    #------------------------------#
    num_classes = 21
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = False
    #------------------------------#
    #   数据集路径
    #------------------------------#
    dataset_path = "VOCdevkit/VOC2007/"

    # 获取model
    model = Unet(inputs_size,num_classes)

    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model_path = "model_data/unet_voc.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
        
    loss = dice_loss_with_CE() if dice_loss else CE()
    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)

    freeze_layers = 17
    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        Batch_size      = 2

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        if eager:
            gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path)
            gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path)
            
            gen             = tf.data.Dataset.from_generator(partial(gen.generate, random_data = True), (tf.float32, tf.float32))
            gen_val         = tf.data.Dataset.from_generator(partial(gen_val.generate, random_data = False), (tf.float32, tf.float32))

            gen             = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val         = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule     = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr, decay_steps=epoch_size, decay_rate=0.92, staircase=True
            )
            optimizer       = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            for epoch in range(Init_Epoch,Freeze_Epoch):
                fit_one_epoch(model, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, get_train_step_fn())
        else:
            gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
            gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path).generate(False)
            
            model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                    optimizer = Adam(lr=lr),
                    metrics = [f_score()])
                    
            model.fit_generator(gen,
                    steps_per_epoch=epoch_size,
                    validation_data=gen_val,
                    validation_steps=epoch_size_val,
                    epochs=Freeze_Epoch,
                    initial_epoch=Init_Epoch,
                    callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])
                    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr              = 1e-5
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100
        Batch_size      = 2

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
            
        if eager:
            gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path)
            gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path)
            
            gen             = tf.data.Dataset.from_generator(partial(gen.generate, random_data = True), (tf.float32, tf.float32))
            gen_val         = tf.data.Dataset.from_generator(partial(gen_val.generate, random_data = False), (tf.float32, tf.float32))

            gen             = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val         = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule     = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr, decay_steps=epoch_size, decay_rate=0.92, staircase=True
            )
            optimizer       = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
                fit_one_epoch(model, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, get_train_step_fn())
        else:
            gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
            gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path).generate(False)

            model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                    optimizer = Adam(lr=lr),
                    metrics = [f_score()])
            
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
            model.fit_generator(gen,
                    steps_per_epoch=epoch_size,
                    validation_data=gen_val,
                    validation_steps=epoch_size_val,
                    epochs=Unfreeze_Epoch,
                    initial_epoch=Freeze_Epoch,
                    callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])

                    
