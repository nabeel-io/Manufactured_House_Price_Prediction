## **Feature Engineering**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### **About Data**


```python
nyc_taxi = pd.read_pickle("data/train.pickle")
nyc_taxi.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2707852</th>
      <td>2</td>
      <td>2022-03-24 21:47:05</td>
      <td>2022-03-24 21:55:22</td>
      <td>1</td>
      <td>1.85</td>
      <td>1</td>
      <td>141</td>
      <td>137</td>
      <td>1</td>
      <td>8.0</td>
      <td>11.80</td>
    </tr>
    <tr>
      <th>2707853</th>
      <td>1</td>
      <td>2022-03-14 10:36:01</td>
      <td>2022-03-14 10:45:09</td>
      <td>1</td>
      <td>1.40</td>
      <td>1</td>
      <td>137</td>
      <td>237</td>
      <td>1</td>
      <td>8.0</td>
      <td>13.55</td>
    </tr>
    <tr>
      <th>2707854</th>
      <td>2</td>
      <td>2022-03-18 09:19:17</td>
      <td>2022-03-18 09:28:55</td>
      <td>1</td>
      <td>1.92</td>
      <td>1</td>
      <td>79</td>
      <td>186</td>
      <td>1</td>
      <td>8.5</td>
      <td>14.16</td>
    </tr>
    <tr>
      <th>2707855</th>
      <td>2</td>
      <td>2022-03-30 11:30:47</td>
      <td>2022-03-30 11:46:04</td>
      <td>1</td>
      <td>1.64</td>
      <td>1</td>
      <td>233</td>
      <td>68</td>
      <td>2</td>
      <td>10.5</td>
      <td>13.80</td>
    </tr>
    <tr>
      <th>2707856</th>
      <td>2</td>
      <td>2022-03-09 14:56:36</td>
      <td>2022-03-09 15:39:52</td>
      <td>1</td>
      <td>4.28</td>
      <td>1</td>
      <td>90</td>
      <td>236</td>
      <td>2</td>
      <td>26.0</td>
      <td>29.30</td>
    </tr>
  </tbody>
</table>
</div>



*Extract day in week from `tpep_pickup_datetime`*


```python
nyc_taxi["day_of_week"] = nyc_taxi["tpep_pickup_datetime"].dt.dayofweek
```

*Calculating time `interval` from pickup to dropoff*


```python
diff = nyc_taxi["tpep_dropoff_datetime"] - nyc_taxi["tpep_pickup_datetime"]
```

*Extracting time interval in `seconds`*


```python
diff = diff.dt.total_seconds()
nyc_taxi["time_interval"] = diff
```

*Extracting `hour` from pickup date time*


```python
nyc_taxi["hour"] = nyc_taxi["tpep_pickup_datetime"].dt.hour
```

*Calculating `distance` for zero values*


```python
pos = nyc_taxi[nyc_taxi["trip_distance"] == 0.0].index
npos = nyc_taxi[~nyc_taxi.index.isin(pos)].index
```


```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(nyc_taxi.loc[npos,["total_amount", "time_interval"]], nyc_taxi.loc[npos, "trip_distance"])
pred = reg.predict(nyc_taxi.loc[pos,["total_amount", "time_interval"]])/2
```


```python
x = 0
for i in pos[:10]:
    nyc_taxi.loc[i, "trip_distance"] = pred[x]
    x = x + 1
```

*Removing negative and zero time instances from time_interval*


```python
pos = nyc_taxi[nyc_taxi["time_interval"] <=0.0].index
nyc_taxi.drop(pos, axis=0, inplace=True)
```

*Converting miles into `meters`*


```python
nyc_taxi["trip_distance"] = nyc_taxi["trip_distance"] * 1609.0
```

*Calculating speed in `meter/second`*


```python
nyc_taxi["avg_speed_ms"] = nyc_taxi["trip_distance"] / nyc_taxi["time_interval"]
```


```python
nyc_taxi.shape
```




    (2705487, 15)




```python
nyc_taxi.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>total_amount</th>
      <th>day_of_week</th>
      <th>time_interval</th>
      <th>hour</th>
      <th>avg_speed_ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2022-03-14 12:06:11</td>
      <td>2022-03-14 12:13:51</td>
      <td>1</td>
      <td>3008.83</td>
      <td>1</td>
      <td>229</td>
      <td>263</td>
      <td>2</td>
      <td>8.0</td>
      <td>11.30</td>
      <td>0</td>
      <td>460.0</td>
      <td>12</td>
      <td>6.540935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2022-03-09 17:56:22</td>
      <td>2022-03-09 18:09:48</td>
      <td>2</td>
      <td>3105.37</td>
      <td>1</td>
      <td>140</td>
      <td>239</td>
      <td>1</td>
      <td>10.0</td>
      <td>15.30</td>
      <td>2</td>
      <td>806.0</td>
      <td>17</td>
      <td>3.852816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2022-03-24 22:11:00</td>
      <td>2022-03-24 22:22:20</td>
      <td>1</td>
      <td>2928.38</td>
      <td>1</td>
      <td>239</td>
      <td>263</td>
      <td>1</td>
      <td>9.5</td>
      <td>15.96</td>
      <td>3</td>
      <td>680.0</td>
      <td>22</td>
      <td>4.306441</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2022-03-15 20:54:50</td>
      <td>2022-03-15 21:04:46</td>
      <td>6</td>
      <td>3137.55</td>
      <td>1</td>
      <td>163</td>
      <td>186</td>
      <td>1</td>
      <td>9.0</td>
      <td>15.36</td>
      <td>1</td>
      <td>596.0</td>
      <td>20</td>
      <td>5.264346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2022-03-27 16:59:01</td>
      <td>2022-03-27 17:10:36</td>
      <td>1</td>
      <td>1657.27</td>
      <td>1</td>
      <td>186</td>
      <td>230</td>
      <td>2</td>
      <td>8.5</td>
      <td>11.80</td>
      <td>6</td>
      <td>695.0</td>
      <td>16</td>
      <td>2.384561</td>
    </tr>
  </tbody>
</table>
</div>



*Removing outliers from columns `trip_distance`, `time_interval` and `total_amount`* 


```python
# removing outliers with more than 4 sd
from scipy import stats
cols = ["trip_distance", "time_interval", "total_amount"]
for col in cols:
    idx = nyc_taxi[np.abs(stats.zscore(nyc_taxi[col]) > 4)].index
    nyc_taxi.drop(idx, axis=0, inplace=True)
```


```python
# save train_labels
train_labels = nyc_taxi["total_amount"]
train_labels = np.array(train_labels)
np.save("data/train_labels", train_labels, allow_pickle=True)
```


```python
del train_labels
```

**Encoding**


```python
from category_encoders import BinaryEncoder
cat = ["VendorID", "passenger_count", "trip_distance", "RatecodeID",
       "PULocationID", "DOLocationID", "payment_type", "day_of_week",
       "hour"]
encoder = BinaryEncoder(cols=cat)
enc = encoder.fit_transform(nyc_taxi.loc[:, cat])
encoded_cat = np.array(enc)
```

### **Scaling** 


```python
from sklearn.preprocessing import StandardScaler
cont = ["trip_distance", "time_interval", "avg_speed_ms"]
scale = StandardScaler()
sca = scale.fit_transform(nyc_taxi.loc[:,cont])
scale_cont = np.array(sca)
```


```python
train_matrix = np.concatenate((encoded_cat, scale_cont), axis=1)
```


```python
np.save(file="data/train_matrix", arr=train_matrix, allow_pickle=True)
```


```python
del nyc_taxi,  scale_cont, encoded_cat, train_matrix 
```

**Test**


```python
test = pd.read_pickle("data/test.pickle")
test.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>899995</th>
      <td>1</td>
      <td>2022-03-31 09:00:44</td>
      <td>2022-03-31 09:26:28</td>
      <td>1</td>
      <td>6.90</td>
      <td>1</td>
      <td>13</td>
      <td>162</td>
      <td>1</td>
      <td>26.0</td>
      <td>35.16</td>
    </tr>
    <tr>
      <th>899996</th>
      <td>2</td>
      <td>2022-03-17 03:14:56</td>
      <td>2022-03-17 03:34:42</td>
      <td>1</td>
      <td>3.84</td>
      <td>1</td>
      <td>170</td>
      <td>87</td>
      <td>2</td>
      <td>16.0</td>
      <td>19.80</td>
    </tr>
    <tr>
      <th>899997</th>
      <td>1</td>
      <td>2022-03-31 13:31:46</td>
      <td>2022-03-31 13:51:19</td>
      <td>0</td>
      <td>1.50</td>
      <td>1</td>
      <td>163</td>
      <td>236</td>
      <td>1</td>
      <td>12.5</td>
      <td>18.95</td>
    </tr>
    <tr>
      <th>899998</th>
      <td>1</td>
      <td>2022-03-28 17:42:44</td>
      <td>2022-03-28 17:58:20</td>
      <td>1</td>
      <td>3.90</td>
      <td>1</td>
      <td>140</td>
      <td>179</td>
      <td>1</td>
      <td>14.5</td>
      <td>23.50</td>
    </tr>
    <tr>
      <th>899999</th>
      <td>2</td>
      <td>2022-03-15 20:24:19</td>
      <td>2022-03-15 20:46:09</td>
      <td>1</td>
      <td>3.92</td>
      <td>1</td>
      <td>148</td>
      <td>48</td>
      <td>1</td>
      <td>17.0</td>
      <td>24.96</td>
    </tr>
  </tbody>
</table>
</div>



*Extract day in week from `tpep_pickup_datetime`*


```python
test["day_of_week"] = test["tpep_pickup_datetime"].dt.dayofweek
```

*Calculating time interval from pickup to dropoff*


```python
diff = test["tpep_dropoff_datetime"] - test["tpep_pickup_datetime"]
```

*Extracting time interval in seconds*


```python
diff = diff.dt.total_seconds()
test["time_interval"] = diff
```

*Extracting hour from pickup date time*


```python
test["hour"] = test["tpep_pickup_datetime"].dt.hour
```

*Calculating distance for zero values*


```python
pos = test[test["trip_distance"] == 0.0].index
npos = test[~test.index.isin(pos)].index
```


```python
pred = reg.predict(test.loc[pos,["total_amount", "time_interval"]])/2
```


```python
x = 0
for i in pos[:10]:
    test.loc[i, "trip_distance"] = pred[x]
    x = x + 1
```

*Removing negative and zero time instances from time_interval*


```python
pos = test[test["time_interval"] <=0.0].index
test.drop(pos, axis=0, inplace=True)
```

*Converting miles into meters*


```python
test["trip_distance"] = test["trip_distance"] * 1609.0
```

*Calculating speed in meter/second*


```python
test["avg_speed_ms"] = test["trip_distance"] / test["time_interval"]
```


```python
# save train_labels
test_labels = test["total_amount"]
test_labels = np.array(test_labels)
np.save("data/test_labels", test_labels, allow_pickle=True)
```


```python
from category_encoders import BinaryEncoder
cat = ["VendorID", "passenger_count", "trip_distance", "RatecodeID",
       "PULocationID", "DOLocationID", "payment_type", "day_of_week",
       "hour"]
enc = encoder.transform(test.loc[:, cat])
encoded_cat = np.array(enc)
```


```python
from sklearn.preprocessing import StandardScaler
cont = ["trip_distance", "time_interval", "avg_speed_ms"]
sca = scale.transform(test.loc[:,cont])
scale_cont = np.array(sca)
```


```python
test_matrix = np.concatenate((encoded_cat, scale_cont), axis=1)
```


```python
np.save(file="data/test_matrix", arr=test_matrix, allow_pickle=True)
```


```python
del encoded_cat, scale_cont, test_matrix
```


```python

```
