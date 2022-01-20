from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from tensorflow.keras import optimizers

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.experimental.enable_tensor_float_32_execution(True)
print("TF32 Enabled:",tf.config.experimental.tensor_float_32_execution_enabled)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

def train():
    model_directory = '../models/EffNet_v3_2/'
    train_data_dir = '../data/EffNet_4graphs_2p/train/'
    validation_data_dir = '../data/EffNet_4graphs_2p/validation/'
    initial_epochs = 10
    Fine_epochs = 100
    img_width, img_height = 600, 600
    train_samples = len(os.listdir(train_data_dir+'long')) + len(os.listdir(train_data_dir+'short'))
    validation_samples = len(os.listdir(validation_data_dir+'long')) + len(os.listdir(validation_data_dir+'short'))
    # 2 classes, long and short
    classes_num = 2
    batch_size = 64  # 128

    conv_base = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    # 1
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap")) #flatten 보다 고효율
    model.add(layers.BatchNormalization())
    # model.add(layers.Flatten(name="flatten"))

    # avoid overfitting
    model.add(layers.Dropout(rate=0.2, name="dropout_out"))
    # 4096개의 노드를 가지는 은닉층
    #model.add(layers.Dense(4096, activation='relu', name="fc1"))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    # Set NUMBER_OF_CLASSES to the number of your final predictions.
    model.add(layers.Dense(classes_num, name='dense_logits')) #2개로 출력하는 출력층
    model.add(layers.Activation('softmax', dtype='float32', name="fc_out"))
    conv_base.trainable = False
    model.summary()
    print("Number of layers:", len(model.layers))

    Nadam = optimizers.Nadam(learning_rate=0.001)#0.0001, decay=1e-6, momentum=0.9
    Nadam = mixed_precision.LossScaleOptimizer(Nadam, dynamic=True)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Nadam,
        metrics=["acc"])

    #컬러 사진이나 복잡한 것은 되도록 rescale을 사용하지 말자
    train_datagen = ImageDataGenerator(#rescale=1.0 / 255,
        horizontal_flip=False)

    test_datagen = ImageDataGenerator(#rescale=1.0 / 255,
        horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical')

    # 체크포인트
    metric = 'val_acc' #acc, acc_loss, val_acc, val_loss
    target_dir = model_directory + "weights-improvement/"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save(model_directory + 'model.h5')
    model.save_weights(model_directory + 'weights.h5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=target_dir + 'weights-improvement-{epoch:02d}-{acc:.2f}.h5',
        monitor=metric, verbose=2, save_best_only=True, mode='auto') 
    #mode="auto" or "min"
    #motior는 무엇을 향상시킬 것인지 정하는 것
    callbacks_list = [checkpoint]

    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=initial_epochs,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=callbacks_list,)

    model.save(model_directory + 'model.h5')
    model.save_weights(model_directory + 'weights.h5')

    ##############################################################
    # 학습과정 시각화
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    ########################################################
    # Fine Tuning
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    """for layer in model.layers[-2:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True # 동결해제

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Nadam,
        metrics=["acc"])

    history_Fine = model.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=initial_epochs + Fine_epochs,
        shuffle=True,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)

    model.save(model_directory + 'fine_model.h5')
    model.save_weights(model_directory + 'fine_weights.h5')
    ###########################################
    # Fine Tuning 시각화
    acc += history_Fine.history['acc']
    val_acc += history_Fine.history['val_acc']

    loss += history_Fine.history['loss']
    val_loss += history_Fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()"""


if __name__ == "__main__":
    train()