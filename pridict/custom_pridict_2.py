import os
import numpy as np
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


img_width, img_height = 300, 300 #407, 405
model_path = '../model/weights-improvement/weights-improvement-104-1.00.h5' ##/model.h5
weights_path = '../model/weights.h5'
model = load_model(model_path)
model.load_weights(weights_path)
test_path = '../data/test'
data_path = '../data'

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    if result[0] > 0.9:
      print("Predicted answer: Long")
      answer = 'long'
      print(result)
      #print(array)
    else:
      print("Predicted answer: Not confident")
      answer = 'n/a'
      print(result)
  else:
    if result[1] > 0.9:
      print("Predicted answer: Short")
      answer = 'short'
      print(result)
    else:
      print("Predicted answer: Not confident")
      answer = 'n/a'
      print(result)

  return answer


tb = 0
ts = 0
fb = 0
fs = 0
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

"""
Check metrics
"""
print("True long: ", tb)
print("True short: ", ts)
print("False long: ", fb)  # important
print("False short: ", fs)
print("No action", na)

precision = (tb+ts) / (tb + ts + fb + fs)
#if(tb+fs != 0):
recall = tb / (tb + fs)
print("Recall: ", recall)
f_measure = (2 * recall * precision) / (recall + precision)
print("F-measure: ", f_measure)
#else:
  #print("Divided by Zero")
print("Precision: ", precision)


