### **Basic Libraries**


```python
import numpy as np
import pandas as pd
```


```python
train = np.load("data/train_matrix.npy")
total_amount = np.load("data/train_labels.npy")
```


```python
print(train.shape)
print(total_amount.shape)
```

    (2689006, 53)
    (2689006,)



```python
test = np.load("data/test_matrix.npy")
test_amount = np.load("data/test_labels.npy")
```

### **Split Training Data into validation and training**


```python
from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(train, total_amount, test_size=0.10, random_state=123)
```


```python
del train, total_amount
```

### **LightGBM without parameters**


```python
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
```


```python
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
reg = LGBMRegressor()
reg.fit(xtrain, ytrain)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMRegressor</label><div class="sk-toggleable__content"><pre>LGBMRegressor()</pre></div></div></div></div></div>




```python
pred = reg.predict(xvalid)
mse = mean_squared_error(yvalid, pred)
rmse = np.sqrt(mse)
print(rmse)
```

    2.7185601578298515



```python
pred = reg.predict(test)
mse = mean_squared_error(test_amount, pred)
rmse = np.sqrt(mse)
print(rmse)
```

    5.702606731198352


### **Parameter optimisation using Optuna**


```python
def optimize(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 400),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "n_jobs": -1
    }
    reg = LGBMRegressor(**params)
    reg.fit(xtrain, ytrain) 
    pred = reg.predict(xvalid)
    mse = mean_squared_error(yvalid, pred)
    rmse = np.sqrt(mse)
    return rmse
```


```python
import optuna
study = optuna.create_study(direction="minimize")
study.optimize(optimize, n_trials=50)
```

    [32m[I 2022-08-21 13:43:40,108][0m A new study created in memory with name: no-name-ab4f0b23-862e-4eb7-9f6f-2c4ea590e271[0m
    [32m[I 2022-08-21 13:43:48,408][0m Trial 0 finished with value: 2.6255510555897037 and parameters: {'max_depth': 15, 'min_child_samples': 337, 'subsample': 0.5012739623603298, 'learning_rate': 0.4222393872703886}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:43:56,797][0m Trial 1 finished with value: 2.6773100940197163 and parameters: {'max_depth': 5, 'min_child_samples': 376, 'subsample': 0.583888054474553, 'learning_rate': 0.6023945798345121}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:44:06,465][0m Trial 2 finished with value: 2.686392565366239 and parameters: {'max_depth': 6, 'min_child_samples': 155, 'subsample': 0.9437615573175333, 'learning_rate': 0.19521353730810825}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:44:16,053][0m Trial 3 finished with value: 2.6597617716256896 and parameters: {'max_depth': 14, 'min_child_samples': 276, 'subsample': 0.7117829750168865, 'learning_rate': 0.2147757542381802}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:44:25,578][0m Trial 4 finished with value: 2.6818938474015948 and parameters: {'max_depth': 5, 'min_child_samples': 159, 'subsample': 0.9404042021340145, 'learning_rate': 0.8512573256320635}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:44:33,990][0m Trial 5 finished with value: 2.675356332592669 and parameters: {'max_depth': 14, 'min_child_samples': 140, 'subsample': 0.6309770415105371, 'learning_rate': 0.9237490684452665}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:44:43,489][0m Trial 6 finished with value: 2.652859798793523 and parameters: {'max_depth': 6, 'min_child_samples': 308, 'subsample': 0.9532755590051474, 'learning_rate': 0.43105290876044833}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:44:57,984][0m Trial 7 finished with value: 2.6272879980107997 and parameters: {'max_depth': 10, 'min_child_samples': 278, 'subsample': 0.722013556163704, 'learning_rate': 0.578817975430647}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:45:11,195][0m Trial 8 finished with value: 2.672971288835499 and parameters: {'max_depth': 5, 'min_child_samples': 46, 'subsample': 0.687145375917938, 'learning_rate': 0.31686490853728766}. Best is trial 0 with value: 2.6255510555897037.[0m
    [32m[I 2022-08-21 13:45:21,377][0m Trial 9 finished with value: 2.6150590267039173 and parameters: {'max_depth': 8, 'min_child_samples': 81, 'subsample': 0.7688895537011109, 'learning_rate': 0.6481747845086222}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:45:30,937][0m Trial 10 finished with value: 2.640028434140096 and parameters: {'max_depth': 9, 'min_child_samples': 73, 'subsample': 0.8222143986672112, 'learning_rate': 0.7556808364402116}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:45:44,900][0m Trial 11 finished with value: 2.8202661151009263 and parameters: {'max_depth': 12, 'min_child_samples': 396, 'subsample': 0.5007717025537513, 'learning_rate': 0.05750968137452073}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:45:54,894][0m Trial 12 finished with value: 2.640570448646413 and parameters: {'max_depth': 8, 'min_child_samples': 221, 'subsample': 0.8108093934747999, 'learning_rate': 0.6844096877059853}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:46:04,572][0m Trial 13 finished with value: 2.618396623430663 and parameters: {'max_depth': 12, 'min_child_samples': 335, 'subsample': 0.8264602210251657, 'learning_rate': 0.47105995765803355}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:46:15,744][0m Trial 14 finished with value: 2.616452700938933 and parameters: {'max_depth': 12, 'min_child_samples': 218, 'subsample': 0.8340570591652692, 'learning_rate': 0.4813680759660257}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:46:25,528][0m Trial 15 finished with value: 2.6281475019627867 and parameters: {'max_depth': 12, 'min_child_samples': 103, 'subsample': 0.8650332520166999, 'learning_rate': 0.7449522656974441}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:46:34,466][0m Trial 16 finished with value: 2.6280237267700595 and parameters: {'max_depth': 8, 'min_child_samples': 213, 'subsample': 0.8757439166297776, 'learning_rate': 0.5805970546795577}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:46:43,665][0m Trial 17 finished with value: 2.6207254156756625 and parameters: {'max_depth': 11, 'min_child_samples': 194, 'subsample': 0.7760702831973019, 'learning_rate': 0.35882212833173854}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:46:52,827][0m Trial 18 finished with value: 2.6499466157682177 and parameters: {'max_depth': 8, 'min_child_samples': 241, 'subsample': 0.9956753719299171, 'learning_rate': 0.8293726493566386}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:01,510][0m Trial 19 finished with value: 2.627436034521212 and parameters: {'max_depth': 10, 'min_child_samples': 111, 'subsample': 0.7564443335644988, 'learning_rate': 0.6696346108093626}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:11,686][0m Trial 20 finished with value: 2.674921870564417 and parameters: {'max_depth': 13, 'min_child_samples': 181, 'subsample': 0.6626738304327142, 'learning_rate': 0.9625279395523378}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:21,172][0m Trial 21 finished with value: 2.616512749978228 and parameters: {'max_depth': 12, 'min_child_samples': 348, 'subsample': 0.8574524433501536, 'learning_rate': 0.5211992313324079}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:30,017][0m Trial 22 finished with value: 2.620649135178257 and parameters: {'max_depth': 11, 'min_child_samples': 267, 'subsample': 0.8816244459001543, 'learning_rate': 0.5040260384492745}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:39,380][0m Trial 23 finished with value: 2.6476632020164974 and parameters: {'max_depth': 9, 'min_child_samples': 359, 'subsample': 0.895626601352321, 'learning_rate': 0.3415304873492696}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:47,977][0m Trial 24 finished with value: 2.627359710771363 and parameters: {'max_depth': 13, 'min_child_samples': 41, 'subsample': 0.795375361143223, 'learning_rate': 0.6364804994985307}. Best is trial 9 with value: 2.6150590267039173.[0m
    [32m[I 2022-08-21 13:47:56,866][0m Trial 25 finished with value: 2.6107093440943974 and parameters: {'max_depth': 11, 'min_child_samples': 113, 'subsample': 0.7656065374492199, 'learning_rate': 0.540359416079188}. Best is trial 25 with value: 2.6107093440943974.[0m
    [32m[I 2022-08-21 13:48:05,534][0m Trial 26 finished with value: 2.613298231930249 and parameters: {'max_depth': 10, 'min_child_samples': 103, 'subsample': 0.7379982141040506, 'learning_rate': 0.5398193465268394}. Best is trial 25 with value: 2.6107093440943974.[0m
    [32m[I 2022-08-21 13:48:14,257][0m Trial 27 finished with value: 2.6427555642032536 and parameters: {'max_depth': 7, 'min_child_samples': 97, 'subsample': 0.6153250971669548, 'learning_rate': 0.7129381362545067}. Best is trial 25 with value: 2.6107093440943974.[0m
    [32m[I 2022-08-21 13:48:23,159][0m Trial 28 finished with value: 2.5923470998961964 and parameters: {'max_depth': 10, 'min_child_samples': 71, 'subsample': 0.7519540546125838, 'learning_rate': 0.556099299351521}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:48:32,297][0m Trial 29 finished with value: 2.645426622760566 and parameters: {'max_depth': 10, 'min_child_samples': 131, 'subsample': 0.734547386235359, 'learning_rate': 0.264336436382164}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:48:41,212][0m Trial 30 finished with value: 2.5988247358260153 and parameters: {'max_depth': 11, 'min_child_samples': 62, 'subsample': 0.6878083638115773, 'learning_rate': 0.4288304176655329}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:48:50,699][0m Trial 31 finished with value: 2.6184718753671916 and parameters: {'max_depth': 11, 'min_child_samples': 66, 'subsample': 0.6832743783717183, 'learning_rate': 0.40870202314454473}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:48:59,905][0m Trial 32 finished with value: 2.6064667204508924 and parameters: {'max_depth': 9, 'min_child_samples': 120, 'subsample': 0.566235187501, 'learning_rate': 0.5626793137842458}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:49:09,785][0m Trial 33 finished with value: 2.6189618715748786 and parameters: {'max_depth': 9, 'min_child_samples': 62, 'subsample': 0.5526403941801059, 'learning_rate': 0.4281589564495932}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:49:19,144][0m Trial 34 finished with value: 2.6010616663743304 and parameters: {'max_depth': 11, 'min_child_samples': 128, 'subsample': 0.542188628845004, 'learning_rate': 0.5649621728475704}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:49:27,980][0m Trial 35 finished with value: 2.6210120122682534 and parameters: {'max_depth': 9, 'min_child_samples': 133, 'subsample': 0.5449425503490908, 'learning_rate': 0.6078370164233795}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:49:38,901][0m Trial 36 finished with value: 2.724905875228392 and parameters: {'max_depth': 10, 'min_child_samples': 169, 'subsample': 0.5656321205734414, 'learning_rate': 0.09925010633557818}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:49:47,765][0m Trial 37 finished with value: 2.6162017541932348 and parameters: {'max_depth': 15, 'min_child_samples': 89, 'subsample': 0.6146675339716113, 'learning_rate': 0.3777906592707897}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:49:57,076][0m Trial 38 finished with value: 2.6322259863861177 and parameters: {'max_depth': 11, 'min_child_samples': 149, 'subsample': 0.5888318995751556, 'learning_rate': 0.2728980048008472}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:05,705][0m Trial 39 finished with value: 2.6575703615608197 and parameters: {'max_depth': 13, 'min_child_samples': 58, 'subsample': 0.5340307094829609, 'learning_rate': 0.8234222790076228}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:14,709][0m Trial 40 finished with value: 2.6124508000840065 and parameters: {'max_depth': 7, 'min_child_samples': 119, 'subsample': 0.6354031946865873, 'learning_rate': 0.5789250887978012}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:23,524][0m Trial 41 finished with value: 2.6156701400467144 and parameters: {'max_depth': 11, 'min_child_samples': 117, 'subsample': 0.6978739463067343, 'learning_rate': 0.5415077352493209}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:32,719][0m Trial 42 finished with value: 2.626801601611011 and parameters: {'max_depth': 10, 'min_child_samples': 83, 'subsample': 0.6520583308375817, 'learning_rate': 0.44907116391814034}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:41,364][0m Trial 43 finished with value: 2.605546558556852 and parameters: {'max_depth': 11, 'min_child_samples': 160, 'subsample': 0.525418895356469, 'learning_rate': 0.5271917357782141}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:49,997][0m Trial 44 finished with value: 2.619167477353031 and parameters: {'max_depth': 9, 'min_child_samples': 165, 'subsample': 0.5203984495846519, 'learning_rate': 0.6139614741231187}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:50:59,794][0m Trial 45 finished with value: 2.6039116385093695 and parameters: {'max_depth': 11, 'min_child_samples': 143, 'subsample': 0.5836051621395755, 'learning_rate': 0.41094730940007146}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:51:09,476][0m Trial 46 finished with value: 2.6595463350569353 and parameters: {'max_depth': 11, 'min_child_samples': 148, 'subsample': 0.6015874747572001, 'learning_rate': 0.18107366425147203}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:51:18,598][0m Trial 47 finished with value: 2.6184230734681675 and parameters: {'max_depth': 12, 'min_child_samples': 194, 'subsample': 0.518068360055788, 'learning_rate': 0.4032667537864855}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:51:27,729][0m Trial 48 finished with value: 2.617098413914966 and parameters: {'max_depth': 13, 'min_child_samples': 78, 'subsample': 0.5741907509278436, 'learning_rate': 0.4760445832425316}. Best is trial 28 with value: 2.5923470998961964.[0m
    [32m[I 2022-08-21 13:51:36,816][0m Trial 49 finished with value: 2.629640249478013 and parameters: {'max_depth': 10, 'min_child_samples': 50, 'subsample': 0.7106819512207351, 'learning_rate': 0.29530254274070633}. Best is trial 28 with value: 2.5923470998961964.[0m



```python
del xtrain, xvalid, ytrain, yvalid
```

### **Final Modal**


```python
train = np.load("data/train_matrix.npy")
total_amount = np.load("data/train_labels.npy")
```


```python
reg = LGBMRegressor(max_depth=10, min_child_samples=13, subsample=0.7, learning_rate=0.25, n_jobs=-1)
reg.fit(train, total_amount)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMRegressor(learning_rate=0.25, max_depth=10, min_child_samples=13,
              subsample=0.7)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMRegressor</label><div class="sk-toggleable__content"><pre>LGBMRegressor(learning_rate=0.25, max_depth=10, min_child_samples=13,
              subsample=0.7)</pre></div></div></div></div></div>



### **Predictions**


```python
pred = reg.predict(test)
mse = mean_squared_error(test_amount, pred)
rmse = np.sqrt(mse)
print(rmse)
```

    5.658210424618493



```python

```
