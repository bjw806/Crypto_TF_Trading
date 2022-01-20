import os
import numpy as np
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 480, 480
model_path = '../model/weights-improvement/weights-improvement-11-0.99.h5'
#weights_path = '../model/weights-improvement/weights-improvement-11-0.99.h5'
model = load_model(model_path)
#model.load_weights(weights_path)
test_path = '../data/test'
data_path = '../data'

np.set_printoptions(precision=5, suppress=True)

def get_max(input_array):
    max_tmp = 0
    for x in range(3):
        if(max_tmp < input_array[x]):
            max_tmp = input_array[x]
            index = x
        else:
            pass
    return index


def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    index = get_max(result)


    if(index == 0):
        if(result[0] > 0.9):
            answer = 'long'
        else:
            answer = 'n/a'
        precision = result[0]
    elif(index == 1):
        if (result[1] > 0.9):
            answer = 'neutral'
        else:
            answer = 'n/a'
        precision = result[1]
    elif (index == 2):
        if (result[2] > 0.9):
            answer = 'short'
        else:
            answer = 'n/a'
        precision = result[2]
    else:
        print("e")
        answer = 'n/a'
        precision = -1

    #precision = f'{precision:0.4f}'
    print(answer, precision)
    return answer


tb = 0
ts = 0
true_neutral = 0
fb = 0
fs = 0
false_neutral = 0
na = 0

for i, ret in enumerate(os.walk(data_path + '/test/long')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: long")
        result = predict(ret[0] + '/' + filename)
        if result == "long":
            tb += 1
        elif result == 'n/a':
            print('no action')
            na += 1
        else:
            fb += 1

for i, ret in enumerate(os.walk(data_path + '/test/short')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: short")
        result = predict(ret[0] + '/' + filename)
        if result == "short":
            ts += 1
        elif result == 'n/a':
            print('no action')
            na += 1
        else:
            fs += 1

for i, ret in enumerate(os.walk(data_path + '/test/neutral')):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        print("Label: neutral")
        result = predict(ret[0] + '/' + filename)
        if result == "neutral":
            true_neutral += 1
        elif result == 'n/a':
            print('no action')
            na += 1
        else:
            false_neutral += 1

"""
Check metrics
"""
print("True long: ", tb)
print("True short: ", ts)
print("True N: ", true_neutral)
print("False long: ", fb)  # important
print("False short: ", fs)
print("False N: ", false_neutral)
print("No action", na)

precision = (tb + ts + true_neutral) / (tb + ts + fb + fs + true_neutral + false_neutral + na)
print("p:", precision)
# if(tb+fs != 0):
recall = tb / (tb + fs)
print("Recall: ", recall)
f_measure = (2 * recall * precision) / (recall + precision)
print("F-measure: ", f_measure)
# else:
# print("Divided by Zero")
print("Precision: ", precision)
