```python
import os
os.chdir('C://Users//leejiwon//Desktop//AIToyProject//train//train')
```


```python
import glob

train_files = glob.glob('C:/Users/leejiwon/Desktop/AIToyProject/train/train/train_00000.npy')
print(train_files)
```

    ['C:/Users/leejiwon/Desktop/AIToyProject/train/train/train_00000.npy']
    


```python
len(train_files)
```




    1




```python
import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, concatenate, Input
from tensorflow.keras import Model

import warnings
warnings.filterwarnings("ignore")
```


```python
def trainGenerator():
    for file in train_files:
        dataset = np.load(file)
        target= dataset[:,:,-1].reshape(120,120,1)
        remove_minus = np.where(target < 0, 0, target)
        feature = dataset[:,:,:4]

        yield (feature, remove_minus)
        
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([120,120,4]),tf.TensorShape([120,120,1])))
train_dataset = train_dataset.batch(256).prefetch(1)
```


```python
color_map = plt.cm.get_cmap('RdBu')
```


```python
color_map = color_map.reversed()
```


```python
image_sample = np.load(train_files[0])
```


```python
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 20))

for i in range(4):
    plt.subplot(1,5,i+1)
    plt.imshow(image_sample[:, :, i], cmap=color_map)

plt.subplot(1,5,5)
plt.imshow(image_sample[:,:,-1], cmap = color_map)
plt.show()
```


![png](output_8_0.png)



```python
def base_model(input_layer, start_neurons):
    
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)

    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation='relu')(uconv1)
    
    return output_layer

input_layer = Input((120, 120, 4))
output_layer = base_model(input_layer,64)
```


```python
model = Model(input_layer, output_layer)
model.compile(loss='mae', optimizer='adam')
model.fit(train_dataset, epochs = 5, verbose=1)
```

    Epoch 1/5
    1/1 [==============================] - 1s 1s/step - loss: 1.6981
    Epoch 2/5
    1/1 [==============================] - 0s 353ms/step - loss: 1.6231
    Epoch 3/5
    1/1 [==============================] - 0s 339ms/step - loss: 1.3758
    Epoch 4/5
    1/1 [==============================] - 0s 322ms/step - loss: 1.3060
    Epoch 5/5
    1/1 [==============================] - 0s 320ms/step - loss: 1.2624
    




    <tensorflow.python.keras.callbacks.History at 0x214deee1588>




```python
test_path = 'C:/Users/leejiwon/Desktop/AIToyProject/test'
test_files = sorted(glob.glob(test_path + '/*.npy'))

X_test = []

for file in tqdm(test_files, desc = 'test'):
    data = np.load(file)
    X_test.append(data)

X_test = np.array(X_test)
```

    test: 100%|██████████| 2674/2674 [00:04<00:00, 606.92it/s]
    


```python
X_test.shape
```




    (2674, 120, 120, 4)




```python
pred = model.predict(X_test)
```


```python
submission = pd.read_csv('C:/Users/leejiwon/Desktop/AIToyProject/data/sample_submission.csv')
```


```python
submission.iloc[:,1:] = pred.reshape(-1, 14400).astype(int)
submission.to_csv(path + '/Dacon_baseline.csv', index = False)
```


```python
path = 'C:/Users/leejiwon/Desktop/AIToyProject/data/'
```


```python

```
