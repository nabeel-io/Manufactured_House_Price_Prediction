# **ANN Model**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
```

### **Import Data**


```python
train = np.load("data/train_matrix.npy")
train_label = np.load("data/train_labels.npy")
```


```python
test = np.load("data/test_matrix.npy")
test_labels = np.load("data/test_labels.npy")
```

### **Split training data into training and validtion set**


```python
from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(train, train_label, test_size=0.10, random_state=123) 
del train, train_label
```

### **Build sequential model**


```python
from tensorflow import keras
```


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = xtrain[0].shape))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 53)                0         
                                                                     
     dense_4 (Dense)             (None, 100)               5400      
                                                                     
     dense_5 (Dense)             (None, 100)               10100     
                                                                     
     dense_6 (Dense)             (None, 100)               10100     
                                                                     
     dense_7 (Dense)             (None, 1)                 101       
                                                                     
    =================================================================
    Total params: 25,701
    Trainable params: 25,701
    Non-trainable params: 0
    _________________________________________________________________



```python
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])
```

### **Train model**


```python
history = model.fit(xtrain, ytrain, 
          batch_size=10000, 
          epochs=75, 
          validation_data=(xvalid, yvalid))
```

    2022-08-21 17:45:26.396552: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 513062260 exceeds 10% of free system memory.


    Epoch 1/75
    243/243 [==============================] - 6s 18ms/step - loss: 35.4691 - root_mean_squared_error: 5.9556 - val_loss: 11.4239 - val_root_mean_squared_error: 3.3799
    Epoch 2/75
    243/243 [==============================] - 4s 17ms/step - loss: 10.0076 - root_mean_squared_error: 3.1635 - val_loss: 11.0860 - val_root_mean_squared_error: 3.3296
    Epoch 3/75
    243/243 [==============================] - 5s 19ms/step - loss: 8.9184 - root_mean_squared_error: 2.9864 - val_loss: 8.8706 - val_root_mean_squared_error: 2.9784
    Epoch 4/75
    243/243 [==============================] - 4s 17ms/step - loss: 8.4936 - root_mean_squared_error: 2.9144 - val_loss: 13.3436 - val_root_mean_squared_error: 3.6529
    Epoch 5/75
    243/243 [==============================] - 4s 17ms/step - loss: 8.3994 - root_mean_squared_error: 2.8982 - val_loss: 8.2089 - val_root_mean_squared_error: 2.8651
    Epoch 6/75
    243/243 [==============================] - 4s 17ms/step - loss: 8.1467 - root_mean_squared_error: 2.8542 - val_loss: 11.0904 - val_root_mean_squared_error: 3.3302
    Epoch 7/75
    243/243 [==============================] - 4s 17ms/step - loss: 8.0138 - root_mean_squared_error: 2.8309 - val_loss: 8.3241 - val_root_mean_squared_error: 2.8851
    Epoch 8/75
    243/243 [==============================] - 4s 17ms/step - loss: 7.7860 - root_mean_squared_error: 2.7903 - val_loss: 14.5251 - val_root_mean_squared_error: 3.8112
    Epoch 9/75
    243/243 [==============================] - 4s 17ms/step - loss: 8.0019 - root_mean_squared_error: 2.8288 - val_loss: 8.2926 - val_root_mean_squared_error: 2.8797
    Epoch 10/75
    243/243 [==============================] - 4s 17ms/step - loss: 7.5650 - root_mean_squared_error: 2.7505 - val_loss: 7.4202 - val_root_mean_squared_error: 2.7240
    Epoch 11/75
    243/243 [==============================] - 4s 18ms/step - loss: 7.4343 - root_mean_squared_error: 2.7266 - val_loss: 12.2687 - val_root_mean_squared_error: 3.5027
    Epoch 12/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.4873 - root_mean_squared_error: 2.7363 - val_loss: 17.2987 - val_root_mean_squared_error: 4.1592
    Epoch 13/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.5060 - root_mean_squared_error: 2.7397 - val_loss: 9.6552 - val_root_mean_squared_error: 3.1073
    Epoch 14/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.4478 - root_mean_squared_error: 2.7291 - val_loss: 8.5409 - val_root_mean_squared_error: 2.9225
    Epoch 15/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.3044 - root_mean_squared_error: 2.7027 - val_loss: 7.5203 - val_root_mean_squared_error: 2.7423
    Epoch 16/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.1420 - root_mean_squared_error: 2.6725 - val_loss: 9.4070 - val_root_mean_squared_error: 3.0671
    Epoch 17/75
    243/243 [==============================] - 5s 20ms/step - loss: 7.1066 - root_mean_squared_error: 2.6658 - val_loss: 7.0722 - val_root_mean_squared_error: 2.6594
    Epoch 18/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.0281 - root_mean_squared_error: 2.6511 - val_loss: 7.0580 - val_root_mean_squared_error: 2.6567
    Epoch 19/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.9831 - root_mean_squared_error: 2.6425 - val_loss: 9.9980 - val_root_mean_squared_error: 3.1620
    Epoch 20/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.0666 - root_mean_squared_error: 2.6583 - val_loss: 8.8886 - val_root_mean_squared_error: 2.9814
    Epoch 21/75
    243/243 [==============================] - 5s 19ms/step - loss: 7.0244 - root_mean_squared_error: 2.6504 - val_loss: 10.1730 - val_root_mean_squared_error: 3.1895
    Epoch 22/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.9058 - root_mean_squared_error: 2.6279 - val_loss: 7.1009 - val_root_mean_squared_error: 2.6648
    Epoch 23/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.7901 - root_mean_squared_error: 2.6058 - val_loss: 7.0551 - val_root_mean_squared_error: 2.6561
    Epoch 24/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.7548 - root_mean_squared_error: 2.5990 - val_loss: 6.9177 - val_root_mean_squared_error: 2.6301
    Epoch 25/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.7127 - root_mean_squared_error: 2.5909 - val_loss: 7.8762 - val_root_mean_squared_error: 2.8065
    Epoch 26/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.6623 - root_mean_squared_error: 2.5811 - val_loss: 6.8943 - val_root_mean_squared_error: 2.6257
    Epoch 27/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.6626 - root_mean_squared_error: 2.5812 - val_loss: 7.8655 - val_root_mean_squared_error: 2.8046
    Epoch 28/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.5947 - root_mean_squared_error: 2.5680 - val_loss: 7.8212 - val_root_mean_squared_error: 2.7966
    Epoch 29/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.6441 - root_mean_squared_error: 2.5776 - val_loss: 7.3907 - val_root_mean_squared_error: 2.7186
    Epoch 30/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.8607 - root_mean_squared_error: 2.6193 - val_loss: 8.7152 - val_root_mean_squared_error: 2.9522
    Epoch 31/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.6782 - root_mean_squared_error: 2.5842 - val_loss: 7.4941 - val_root_mean_squared_error: 2.7375
    Epoch 32/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.5116 - root_mean_squared_error: 2.5518 - val_loss: 6.8758 - val_root_mean_squared_error: 2.6222
    Epoch 33/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.4476 - root_mean_squared_error: 2.5392 - val_loss: 7.1412 - val_root_mean_squared_error: 2.6723
    Epoch 34/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.5435 - root_mean_squared_error: 2.5580 - val_loss: 9.7364 - val_root_mean_squared_error: 3.1203
    Epoch 35/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.4808 - root_mean_squared_error: 2.5457 - val_loss: 6.8344 - val_root_mean_squared_error: 2.6143
    Epoch 36/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.4226 - root_mean_squared_error: 2.5343 - val_loss: 6.9098 - val_root_mean_squared_error: 2.6286
    Epoch 37/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.3722 - root_mean_squared_error: 2.5243 - val_loss: 12.5302 - val_root_mean_squared_error: 3.5398
    Epoch 38/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.6527 - root_mean_squared_error: 2.5793 - val_loss: 6.9013 - val_root_mean_squared_error: 2.6270
    Epoch 39/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.3827 - root_mean_squared_error: 2.5264 - val_loss: 6.8630 - val_root_mean_squared_error: 2.6197
    Epoch 40/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.3624 - root_mean_squared_error: 2.5224 - val_loss: 7.6118 - val_root_mean_squared_error: 2.7589
    Epoch 41/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.5543 - root_mean_squared_error: 2.5601 - val_loss: 7.0830 - val_root_mean_squared_error: 2.6614
    Epoch 42/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2606 - root_mean_squared_error: 2.5021 - val_loss: 6.8418 - val_root_mean_squared_error: 2.6157
    Epoch 43/75
    243/243 [==============================] - 5s 20ms/step - loss: 6.3892 - root_mean_squared_error: 2.5277 - val_loss: 11.0380 - val_root_mean_squared_error: 3.3224
    Epoch 44/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.4675 - root_mean_squared_error: 2.5431 - val_loss: 7.1026 - val_root_mean_squared_error: 2.6651
    Epoch 45/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2344 - root_mean_squared_error: 2.4969 - val_loss: 7.6271 - val_root_mean_squared_error: 2.7617
    Epoch 46/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2823 - root_mean_squared_error: 2.5065 - val_loss: 6.9287 - val_root_mean_squared_error: 2.6322
    Epoch 47/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.3203 - root_mean_squared_error: 2.5140 - val_loss: 8.9484 - val_root_mean_squared_error: 2.9914
    Epoch 48/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2873 - root_mean_squared_error: 2.5074 - val_loss: 6.6007 - val_root_mean_squared_error: 2.5692
    Epoch 49/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1382 - root_mean_squared_error: 2.4775 - val_loss: 7.5195 - val_root_mean_squared_error: 2.7422
    Epoch 50/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1455 - root_mean_squared_error: 2.4790 - val_loss: 6.6136 - val_root_mean_squared_error: 2.5717
    Epoch 51/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1203 - root_mean_squared_error: 2.4739 - val_loss: 6.7955 - val_root_mean_squared_error: 2.6068
    Epoch 52/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2446 - root_mean_squared_error: 2.4989 - val_loss: 6.8106 - val_root_mean_squared_error: 2.6097
    Epoch 53/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1838 - root_mean_squared_error: 2.4867 - val_loss: 6.6165 - val_root_mean_squared_error: 2.5723
    Epoch 54/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1001 - root_mean_squared_error: 2.4698 - val_loss: 6.9552 - val_root_mean_squared_error: 2.6373
    Epoch 55/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1001 - root_mean_squared_error: 2.4698 - val_loss: 9.5169 - val_root_mean_squared_error: 3.0849
    Epoch 56/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2258 - root_mean_squared_error: 2.4952 - val_loss: 6.6981 - val_root_mean_squared_error: 2.5881
    Epoch 57/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.0091 - root_mean_squared_error: 2.4513 - val_loss: 9.6090 - val_root_mean_squared_error: 3.0998
    Epoch 58/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.0826 - root_mean_squared_error: 2.4663 - val_loss: 6.8233 - val_root_mean_squared_error: 2.6121
    Epoch 59/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9665 - root_mean_squared_error: 2.4426 - val_loss: 7.5699 - val_root_mean_squared_error: 2.7513
    Epoch 60/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1967 - root_mean_squared_error: 2.4893 - val_loss: 8.2670 - val_root_mean_squared_error: 2.8752
    Epoch 61/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.0098 - root_mean_squared_error: 2.4515 - val_loss: 6.9415 - val_root_mean_squared_error: 2.6347
    Epoch 62/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9493 - root_mean_squared_error: 2.4391 - val_loss: 6.6441 - val_root_mean_squared_error: 2.5776
    Epoch 63/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9698 - root_mean_squared_error: 2.4433 - val_loss: 6.7441 - val_root_mean_squared_error: 2.5969
    Epoch 64/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9805 - root_mean_squared_error: 2.4455 - val_loss: 6.7858 - val_root_mean_squared_error: 2.6050
    Epoch 65/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.0095 - root_mean_squared_error: 2.4514 - val_loss: 6.9147 - val_root_mean_squared_error: 2.6296
    Epoch 66/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2247 - root_mean_squared_error: 2.4949 - val_loss: 6.6169 - val_root_mean_squared_error: 2.5723
    Epoch 67/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9821 - root_mean_squared_error: 2.4458 - val_loss: 7.0259 - val_root_mean_squared_error: 2.6506
    Epoch 68/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9720 - root_mean_squared_error: 2.4438 - val_loss: 8.2971 - val_root_mean_squared_error: 2.8805
    Epoch 69/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9399 - root_mean_squared_error: 2.4372 - val_loss: 7.1006 - val_root_mean_squared_error: 2.6647
    Epoch 70/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9047 - root_mean_squared_error: 2.4299 - val_loss: 8.5469 - val_root_mean_squared_error: 2.9235
    Epoch 71/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.1483 - root_mean_squared_error: 2.4796 - val_loss: 8.1744 - val_root_mean_squared_error: 2.8591
    Epoch 72/75
    243/243 [==============================] - 5s 19ms/step - loss: 6.2185 - root_mean_squared_error: 2.4937 - val_loss: 8.3760 - val_root_mean_squared_error: 2.8941
    Epoch 73/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9253 - root_mean_squared_error: 2.4342 - val_loss: 6.4739 - val_root_mean_squared_error: 2.5444
    Epoch 74/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.8135 - root_mean_squared_error: 2.4111 - val_loss: 6.7686 - val_root_mean_squared_error: 2.6017
    Epoch 75/75
    243/243 [==============================] - 5s 19ms/step - loss: 5.9616 - root_mean_squared_error: 2.4416 - val_loss: 6.6393 - val_root_mean_squared_error: 2.5767


### **Loss**


```python
loss = pd.DataFrame(history.history)
loss.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>root_mean_squared_error</th>
      <th>val_loss</th>
      <th>val_root_mean_squared_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>6.148306</td>
      <td>2.479578</td>
      <td>8.174443</td>
      <td>2.859098</td>
    </tr>
    <tr>
      <th>71</th>
      <td>6.218507</td>
      <td>2.493693</td>
      <td>8.375956</td>
      <td>2.894124</td>
    </tr>
    <tr>
      <th>72</th>
      <td>5.925297</td>
      <td>2.434193</td>
      <td>6.473879</td>
      <td>2.544382</td>
    </tr>
    <tr>
      <th>73</th>
      <td>5.813494</td>
      <td>2.411119</td>
      <td>6.768629</td>
      <td>2.601659</td>
    </tr>
    <tr>
      <th>74</th>
      <td>5.961595</td>
      <td>2.441638</td>
      <td>6.639330</td>
      <td>2.576690</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(nrows=2,figsize=(8,5), dpi=100)
sns.lineplot(data=loss.loc[:,["loss","val_loss"]], palette="RdPu",linewidth=1.0, ax=axes[0]);
sns.lineplot(data=loss.loc[:,["root_mean_squared_error","val_root_mean_squared_error"]], palette="BuGn",linewidth=1.0, ax=axes[1]);
```


![png](output_16_0.png)


### **Prediction**


```python
model.evaluate(test, test_labels)
```


    28100/28100 [==============================] - 23s 830us/step - loss: 33.0765 - root_mean_squared_error: 5.7512

    [33.076515197753906, 5.751218795776367]




```python

```
