from tensorflow.keras.applications import *  # Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from tensorflow.keras import optimizers
import keras_efficientnet_v2

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.experimental.enable_tensor_float_32_execution(True)
print("TF32 Enabled:", tf.config.experimental.tensor_float_32_execution_enabled)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


def train():
    model_directory = '../model/'
    train_data_dir = '../data/train/'
    validation_data_dir = '../data/validation/'
    initial_epochs = 10
    fine_epochs = 5
    img_width, img_height = 480, 480
    train_samples = len(os.listdir(train_data_dir + 'long')) + len(os.listdir(train_data_dir + 'short')) + len(os.listdir(train_data_dir + 'neutral'))
    validation_samples = len(os.listdir(validation_data_dir + 'long')) + len(os.listdir(validation_data_dir + 'short')) + len(os.listdir(validation_data_dir + 'neutral'))
    classes_num = 3  # 2 classes, long and short
    batch_size = 4  # 128
    metric = 'acc'  # acc, acc_loss, val_acc, val_loss
    fine_metric = 'val_acc'

    base = keras_efficientnet_v2.EfficientNetV2L(input_shape=(img_width, img_height, 3), dropout=1e-6, num_classes=0, pretrained=None, include_preprocessing=False)

    base.trainable = True
    model = models.Sequential()
    model.add(base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))  # flatten 보다 고효율
    model.add(layers.Dense(classes_num, name='dense_logits'))  # 2개로 출력하는 출력층
    model.add(layers.Activation('softmax', dtype='float32', name="fc_out"))
    model.summary()

    Nadam = optimizers.Nadam(learning_rate=0.0001)  # 0.0001, decay=1e-6, momentum=0.9
    Nadam = mixed_precision.LossScaleOptimizer(Nadam, dynamic=True)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Nadam,
        metrics=["acc"])

    train_datagen = ImageDataGenerator(  # rescale=1.0 / 255,
        horizontal_flip=False)

    test_datagen = ImageDataGenerator(  # rescale=1.0 / 255,
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

    target_dir = model_directory + "weights-improvement/"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save(model_directory + 'model.h5')
    model.save_weights(model_directory + 'weights.h5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=target_dir + 'weights-improvement-{epoch:02d}-{acc:.2f}.h5',
        monitor=metric, verbose=2, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    history = model.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=initial_epochs,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=callbacks_list)

    model.save(model_directory + 'model.h5')
    model.save_weights(model_directory + 'weights.h5')

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
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    base.trainable = True

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Nadam,
        metrics=["acc"])

    #####################
    fine_target_dir = model_directory + "fine-improvement/"
    if not os.path.exists(fine_target_dir):
        os.mkdir(fine_target_dir)
    model.save(model_directory + 'fine_model.h5')
    model.save_weights(model_directory + 'fine_weights.h5')

    fine_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=fine_target_dir + 'fine-improvement-{epoch:02d}-{acc:.2f}.h5',
        monitor=fine_metric, verbose=2, save_best_only=True, mode='auto')
    fine_callbacks_list = [fine_checkpoint]

    history_Fine = model.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=initial_epochs + fine_epochs,
        shuffle=True,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=fine_callbacks_list)

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
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == "__main__":
    train()