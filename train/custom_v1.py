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
    model_dir_name = 'EffNetV1_V1'
    data_dir_name = 'EffNet_2graphs'
    initial_epochs = 10
    img_width, img_height = 300, 300
    batch_size = 64
    metric = 'acc' #acc, acc_loss, val_acc, val_loss

    model_directory = '../models/' + model_dir_name
    train_data_dir = '../data/' + data_dir_name + '/train/'
    validation_data_dir = '../data/'+ data_dir_name +'/validation/'
    train_samples = len(os.listdir(train_data_dir+'long')) + len(os.listdir(train_data_dir+'short'))
    validation_samples = len(os.listdir(validation_data_dir+'long')) + len(os.listdir(validation_data_dir+'short'))
    classes_num = 2 # 2 classes, long and short

    model = models.Sequential()
    model.add(layers.Conv2D(32, (15,15), activation="relu", input_shape=(img_width, img_height , 3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    # 레이어 2 
    model.add(layers.Conv2D(64, (11,11), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    # 레이어3
    model.add(layers.Conv2D(128, (9,9), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    # 레이어4
    model.add(layers.Conv2D(256, (7,7), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    # 레이어5
    model.add(layers.Conv2D(512, (5,5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    # 레이어6
    model.add(layers.Conv2D(1024, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    # Fully Connected 
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes_num, name='dense_logits')) #2개로 출력하는 출력층
    model.add(layers.Activation('softmax', dtype='float32', name="fc_out"))
    model.summary()

    Nadam = optimizers.Nadam(learning_rate=0.001)#0.0001, decay=1e-6, momentum=0.9
    Nadam = mixed_precision.LossScaleOptimizer(Nadam, dynamic=True)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Nadam,
        metrics=["acc"])

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
    target_dir = model_directory + "checkpoint/"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save(model_directory + 'model.h5')
    model.save_weights(model_directory + 'weights.h5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=target_dir + 'checkpoint-{epoch:02d}-acc{acc:.2f}.h5',
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

if __name__ == "__main__":
    train()