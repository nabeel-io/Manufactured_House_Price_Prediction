## **Data Cleaning**

 **Importing basic libraries**


```python
import numpy as np
import pandas as pd
```

**Importing data**


```python
# importing data
nyc_taxi = pd.read_parquet("data/yellow_tripdata_2022-03.parquet")
```

**Removing non essential features**

*In our analysis `store_and_fwd_flag`, `fare_amount`, `extra` , `mta_tax`, `tip_amount` ,`tolls_amount` ,
`improvement_surcharge`, `congestion_surcharge`, `airport_fee` are dropped.*


```python
drop_cols = ["store_and_fwd_flag", "extra" , "mta_tax", "tip_amount","tolls_amount" , 
             "improvement_surcharge", "congestion_surcharge", "airport_fee"]
nyc_taxi.drop(drop_cols, axis=1,inplace=True)
```

### **Missing Values**


```python
nyc_taxi.isna().sum()
```




    VendorID                      0
    tpep_pickup_datetime          0
    tpep_dropoff_datetime         0
    passenger_count          117814
    trip_distance                 0
    RatecodeID               117814
    PULocationID                  0
    DOLocationID                  0
    payment_type                  0
    fare_amount                   0
    total_amount                  0
    dtype: int64



*There are `117814` values are missing from column `passenger_count` and `Rate_Code_ID`*

**Dealing with RatecodeID**

*Majority of instances belong to category 1 . We will impute all `missing` values with `category 1`*


```python
nyc_taxi["RatecodeID"].value_counts()/len(nyc_taxi)
```




    1.0     0.921587
    2.0     0.033114
    5.0     0.005875
    99.0    0.003550
    3.0     0.002237
    4.0     0.001156
    6.0     0.000007
    Name: RatecodeID, dtype: float64




```python
miss_pos = nyc_taxi[nyc_taxi["RatecodeID"].isna()].index
nyc_taxi.loc[miss_pos, "RatecodeID"] = 1.0
```

**Dealing with passenger_count**


```python
np.round(nyc_taxi["passenger_count"].value_counts()/len(nyc_taxi),2)
```




    1.0    0.72
    2.0    0.14
    3.0    0.04
    0.0    0.02
    5.0    0.02
    4.0    0.01
    6.0    0.01
    7.0    0.00
    8.0    0.00
    9.0    0.00
    Name: passenger_count, dtype: float64



*Impute all missing class with  `category 1` as it is the majority class in most instances*


```python
nyc_taxi.loc[nyc_taxi[nyc_taxi["passenger_count"].isna()].index, "passenger_count"] = 1.0
```

*All missing instances are now filled with appropriate value*


```python
nyc_taxi.isna().sum()
```




    VendorID                 0
    tpep_pickup_datetime     0
    tpep_dropoff_datetime    0
    passenger_count          0
    trip_distance            0
    RatecodeID               0
    PULocationID             0
    DOLocationID             0
    payment_type             0
    fare_amount              0
    total_amount             0
    dtype: int64



### **Data Type**


```python
nyc_taxi.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3627882 entries, 0 to 3627881
    Data columns (total 11 columns):
     #   Column                 Dtype         
    ---  ------                 -----         
     0   VendorID               int64         
     1   tpep_pickup_datetime   datetime64[ns]
     2   tpep_dropoff_datetime  datetime64[ns]
     3   passenger_count        float64       
     4   trip_distance          float64       
     5   RatecodeID             float64       
     6   PULocationID           int64         
     7   DOLocationID           int64         
     8   payment_type           int64         
     9   fare_amount            float64       
     10  total_amount           float64       
    dtypes: datetime64[ns](2), float64(5), int64(4)
    memory usage: 304.5 MB


*Since `passenger_count` and `RatecodeID` are categorical variables we will change there data type from float to int*


```python
nyc_taxi["passenger_count"] = nyc_taxi["passenger_count"].astype("int")
nyc_taxi["RatecodeID"] = nyc_taxi["RatecodeID"].astype("int")
```

*Convert the `int64` datatype to `int16` to save memory space*


```python
for col in nyc_taxi.columns:
    if nyc_taxi[col].dtype == "int64":
        nyc_taxi[col] = nyc_taxi[col].astype("int16")
    else:
        pass
```

*We don't convert the `float64` because it needs `high precision`*


```python
# required format
nyc_taxi.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3627882 entries, 0 to 3627881
    Data columns (total 11 columns):
     #   Column                 Dtype         
    ---  ------                 -----         
     0   VendorID               int16         
     1   tpep_pickup_datetime   datetime64[ns]
     2   tpep_dropoff_datetime  datetime64[ns]
     3   passenger_count        int16         
     4   trip_distance          float64       
     5   RatecodeID             int16         
     6   PULocationID           int16         
     7   DOLocationID           int16         
     8   payment_type           int16         
     9   fare_amount            float64       
     10  total_amount           float64       
    dtypes: datetime64[ns](2), float64(3), int16(6)
    memory usage: 179.9 MB


*Reduce the size of dataframe by `41%`*

### **Saving the dataframe in pickle file**


```python
nyc_taxi.to_pickle("data/nyc_taxi.pickle")
```


```python
nyc_taxi.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
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
      <th>count</th>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
      <td>3.627882e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.714948e+00</td>
      <td>1.376396e+00</td>
      <td>5.761290e+00</td>
      <td>1.412461e+00</td>
      <td>1.649635e+02</td>
      <td>1.630534e+02</td>
      <td>1.180307e+00</td>
      <td>1.393749e+01</td>
      <td>2.059364e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.984502e-01</td>
      <td>9.596307e-01</td>
      <td>5.694616e+02</td>
      <td>5.836790e+00</td>
      <td>6.503559e+01</td>
      <td>6.997796e+01</td>
      <td>4.971751e-01</td>
      <td>1.320369e+01</td>
      <td>1.653309e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-8.950000e+02</td>
      <td>-8.953000e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.100000e+00</td>
      <td>1.000000e+00</td>
      <td>1.320000e+02</td>
      <td>1.130000e+02</td>
      <td>1.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.184000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.830000e+00</td>
      <td>1.000000e+00</td>
      <td>1.620000e+02</td>
      <td>1.620000e+02</td>
      <td>1.000000e+00</td>
      <td>1.000000e+01</td>
      <td>1.536000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000e+00</td>
      <td>1.000000e+00</td>
      <td>3.400000e+00</td>
      <td>1.000000e+00</td>
      <td>2.340000e+02</td>
      <td>2.340000e+02</td>
      <td>1.000000e+00</td>
      <td>1.550000e+01</td>
      <td>2.182000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000e+00</td>
      <td>9.000000e+00</td>
      <td>2.862598e+05</td>
      <td>9.900000e+01</td>
      <td>2.650000e+02</td>
      <td>2.650000e+02</td>
      <td>5.000000e+00</td>
      <td>1.777000e+03</td>
      <td>1.783850e+03</td>
    </tr>
  </tbody>
</table>
</div>



### **Correction made after EDA**


```python
nyc_taxi = pd.read_pickle("data/nyc_taxi.pickle")
```

*Replacing RateCodeID with 99 classification to 1 (the majority value)* 


```python
nyc_taxi["RatecodeID"].replace(99, 1, inplace=True)
```

*Removing instances with negative total_amount*


```python
pos = nyc_taxi[nyc_taxi["total_amount"] < 0].index
nyc_taxi.drop(index=pos, axis=0, inplace=True)
```

### **Split test and training sets**


```python
# shuffle
nyc_taxi = nyc_taxi.sample(frac=1.0)
```


```python
# 25% for test and 75% for train
test = nyc_taxi.iloc[:900000, ]
train = nyc_taxi.iloc[900000:, ]
```


```python
test.reset_index(drop=True).to_pickle("data/test.pickle")
train.reset_index(drop=True).to_pickle("data/train.pickle")
```


```python

```
