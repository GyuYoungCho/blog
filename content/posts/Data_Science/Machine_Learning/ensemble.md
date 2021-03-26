---
title: "Ensemble"
date: 2021-03-26T14:35:41+09:00
draft: false
categories : ["Data_Science", "Machine_Learning"]
---

`Randomforest의 데이터 활용


```python
feature_columns = list(data.columns.difference(['target'])) 
X = data[feature_columns] 
y = after_mapping_target
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)  
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 
```

    (49502, 93) (12376, 93) (49502,) (12376,)
    

## 1. XGBoost


```python
# !pip install xgboost
import xgboost as xgb
import time
start = time.time() 
xgb_dtrain = xgb.DMatrix(data = train_x, label = train_y) # 학습 데이터를 XGBoost 모델에 맞게 변환
xgb_dtest = xgb.DMatrix(data = test_x) # 평가 데이터를 XGBoost 모델에 맞게 변환
xgb_param = {'max_depth': 10, # 트리 깊이
         'learning_rate': 0.01, # Step Size
         'n_estimators': 100, # Number of trees, 트리 생성 개수
         'objective': 'multi:softmax', # 목적 함수
        'num_class': len(set(train_y)) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
xgb_model = xgb.train(params = xgb_param, dtrain = xgb_dtrain) # 학습 진행
xgb_model_predict = xgb_model.predict(xgb_dtest) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, xgb_model_predict) * 100), "%") # 정확도 % 계산
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산
```

    Accuracy: 76.67 %
    Time: 6.35 seconds
    


```python
xgb_model_predict
```




    array([5., 3., 6., ..., 9., 2., 7.], dtype=float32)



## 2. LightGBM


```python
# !pip install lightgbm
import lightgbm as lgb
start = time.time() # 시작 시간 지정
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 100, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(train_y)) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
lgb_model_predict = np.argmax(lgb_model.predict(test_x), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, lgb_model_predict) * 100), "%") # 정확도 % 계산
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산
```


    Accuracy: 73.57 %
    Time: 4.08 seconds



## 3. Catboost


```python
# !pip install catboost
import catboost as cb
start = time.time() # 시작 시간 지정
cb_dtrain = cb.Pool(data = train_x, label = train_y) # 학습 데이터를 Catboost 모델에 맞게 변환
cb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 100, # Number of trees, 트리 생성 개수
            'eval_metric': 'Accuracy', # 평가 척도
            'loss_function': 'MultiClass'} # 손실 함수, 목적 함수
cb_model = cb.train(pool = cb_dtrain, params = cb_param) # 학습 진행
cb_model_predict = np.argmax(cb_model.predict(test_x), axis = 1) + 1 # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측, 인덱스의 순서를 맞추기 위해 +1
print("Accuracy: %.2f" % (accuracy_score(test_y, cb_model_predict) * 100), "%") # 정확도 % 계산
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산
```

    0:	learn: 0.6114702	total: 412ms	remaining: 40.8s
    1:	learn: 0.6235304	total: 760ms	remaining: 37.2s
    2:	learn: 0.6410246	total: 1.12s	remaining: 36.1s
    3:	learn: 0.6410650	total: 1.47s	remaining: 35.3s
    4:	learn: 0.6442770	total: 1.81s	remaining: 34.5s
    5:	learn: 0.6455901	total: 2.18s	remaining: 34.1s
    6:	learn: 0.6468224	total: 2.55s	remaining: 33.8s
    7:	learn: 0.6484991	total: 2.88s	remaining: 33.2s
    8:	learn: 0.6512666	total: 3.23s	remaining: 32.6s
    9:	learn: 0.6491859	total: 3.56s	remaining: 32s
    10:	learn: 0.6517716	total: 3.9s	remaining: 31.6s
    11:	learn: 0.6526807	total: 4.25s	remaining: 31.2s
    12:	learn: 0.6531655	total: 4.59s	remaining: 30.8s
    13:	learn: 0.6537918	total: 4.93s	remaining: 30.3s
    14:	learn: 0.6566603	total: 5.25s	remaining: 29.8s
    15:	learn: 0.6563573	total: 5.6s	remaining: 29.4s
    16:	learn: 0.6582966	total: 5.93s	remaining: 29s
    17:	learn: 0.6591047	total: 6.27s	remaining: 28.6s
    18:	learn: 0.6597309	total: 6.62s	remaining: 28.2s
    19:	learn: 0.6603572	total: 7s	remaining: 28s
    20:	learn: 0.6593067	total: 7.38s	remaining: 27.8s
    21:	learn: 0.6603976	total: 7.77s	remaining: 27.5s
    22:	learn: 0.6602966	total: 8.13s	remaining: 27.2s
    23:	learn: 0.6606804	total: 8.48s	remaining: 26.8s
    24:	learn: 0.6612258	total: 8.85s	remaining: 26.5s
    25:	learn: 0.6604178	total: 9.21s	remaining: 26.2s
    26:	learn: 0.6616702	total: 9.57s	remaining: 25.9s
    27:	learn: 0.6636904	total: 9.91s	remaining: 25.5s
    28:	learn: 0.6640540	total: 10.3s	remaining: 25.1s
    29:	learn: 0.6648620	total: 10.6s	remaining: 24.7s
    30:	learn: 0.6652660	total: 10.9s	remaining: 24.4s
    31:	learn: 0.6653469	total: 11.3s	remaining: 24s
    32:	learn: 0.6663165	total: 11.6s	remaining: 23.6s
    33:	learn: 0.6674478	total: 11.9s	remaining: 23.2s
    34:	learn: 0.6675084	total: 12.3s	remaining: 22.8s
    35:	learn: 0.6675488	total: 12.6s	remaining: 22.4s
    36:	learn: 0.6690033	total: 12.9s	remaining: 22s
    37:	learn: 0.6692659	total: 13.3s	remaining: 21.7s
    38:	learn: 0.6702759	total: 13.6s	remaining: 21.3s
    39:	learn: 0.6704982	total: 14s	remaining: 20.9s
    40:	learn: 0.6707608	total: 14.3s	remaining: 20.6s
    41:	learn: 0.6712052	total: 14.6s	remaining: 20.2s
    42:	learn: 0.6714274	total: 15s	remaining: 19.8s
    43:	learn: 0.6715082	total: 15.3s	remaining: 19.5s
    44:	learn: 0.6715890	total: 15.6s	remaining: 19.1s
    45:	learn: 0.6718112	total: 16s	remaining: 18.7s
    46:	learn: 0.6716900	total: 16.3s	remaining: 18.4s
    47:	learn: 0.6719324	total: 16.6s	remaining: 18s
    48:	learn: 0.6730839	total: 17s	remaining: 17.6s
    49:	learn: 0.6735889	total: 17.3s	remaining: 17.3s
    50:	learn: 0.6738314	total: 17.6s	remaining: 16.9s
    51:	learn: 0.6740940	total: 18s	remaining: 16.6s
    52:	learn: 0.6746192	total: 18.3s	remaining: 16.2s
    53:	learn: 0.6750434	total: 18.6s	remaining: 15.9s
    54:	learn: 0.6750636	total: 19s	remaining: 15.5s
    55:	learn: 0.6753465	total: 19.3s	remaining: 15.2s
    56:	learn: 0.6760939	total: 19.6s	remaining: 14.8s
    57:	learn: 0.6768615	total: 20s	remaining: 14.5s
    58:	learn: 0.6769019	total: 20.3s	remaining: 14.1s
    59:	learn: 0.6777908	total: 20.6s	remaining: 13.8s
    60:	learn: 0.6780534	total: 21s	remaining: 13.4s
    61:	learn: 0.6791241	total: 21.3s	remaining: 13.1s
    62:	learn: 0.6796089	total: 21.7s	remaining: 12.7s
    63:	learn: 0.6798715	total: 22s	remaining: 12.4s
    64:	learn: 0.6802351	total: 22.3s	remaining: 12s
    65:	learn: 0.6805180	total: 22.6s	remaining: 11.7s
    66:	learn: 0.6803361	total: 23s	remaining: 11.3s
    67:	learn: 0.6812048	total: 23.3s	remaining: 11s
    68:	learn: 0.6817906	total: 23.6s	remaining: 10.6s
    69:	learn: 0.6819320	total: 24s	remaining: 10.3s
    70:	learn: 0.6827805	total: 24.3s	remaining: 9.92s
    71:	learn: 0.6829219	total: 24.6s	remaining: 9.58s
    72:	learn: 0.6827401	total: 25s	remaining: 9.23s
    73:	learn: 0.6831239	total: 25.3s	remaining: 8.89s
    74:	learn: 0.6833057	total: 25.6s	remaining: 8.54s
    75:	learn: 0.6841340	total: 26s	remaining: 8.2s
    76:	learn: 0.6846390	total: 26.3s	remaining: 7.87s
    77:	learn: 0.6854471	total: 26.7s	remaining: 7.53s
    78:	learn: 0.6862551	total: 27s	remaining: 7.18s
    79:	learn: 0.6872450	total: 27.4s	remaining: 6.84s
    80:	learn: 0.6875278	total: 27.7s	remaining: 6.5s
    81:	learn: 0.6883560	total: 28s	remaining: 6.15s
    82:	learn: 0.6879722	total: 28.4s	remaining: 5.81s
    83:	learn: 0.6887196	total: 28.7s	remaining: 5.47s
    84:	learn: 0.6889419	total: 29s	remaining: 5.12s
    85:	learn: 0.6893459	total: 29.4s	remaining: 4.78s
    86:	learn: 0.6894671	total: 29.7s	remaining: 4.44s
    87:	learn: 0.6896691	total: 30s	remaining: 4.09s
    88:	learn: 0.6900327	total: 30.4s	remaining: 3.75s
    89:	learn: 0.6903963	total: 30.7s	remaining: 3.41s
    90:	learn: 0.6911236	total: 31s	remaining: 3.07s
    91:	learn: 0.6913054	total: 31.4s	remaining: 2.73s
    92:	learn: 0.6913458	total: 31.7s	remaining: 2.39s
    93:	learn: 0.6919720	total: 32s	remaining: 2.04s
    94:	learn: 0.6920326	total: 32.4s	remaining: 1.7s
    95:	learn: 0.6926185	total: 32.7s	remaining: 1.36s
    96:	learn: 0.6929619	total: 33s	remaining: 1.02s
    97:	learn: 0.6932245	total: 33.3s	remaining: 681ms
    98:	learn: 0.6937093	total: 33.7s	remaining: 340ms
    99:	learn: 0.6940124	total: 34s	remaining: 0us
    Accuracy: 69.49 %
    Time: 34.71 seconds
    

---

## Ensemble의 Ensemble


```python
import random
bagging_predict_result = [] # 빈 리스트 생성
for _ in range(10):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    lgb_dtrain = lgb.Dataset(data = train_x.iloc[random_data_index,], label = train_y.iloc[random_data_index,]) # 학습 데이터를 LightGBM 모델에 맞게 변환
    lgb_param = {'max_depth': 14, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 500, # Number of trees, 트리 생성 개수
            'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
    lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
    predict1 = lgb_model.predict(test_x) # 테스트 데이터 예측
    bagging_predict_result.append(predict1) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장
```

    9615
    

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:118: UserWarning: Found `n_estimators` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

    9602
    9601
    9563
    9468
    9509
    9601
    9571
    9480
    9640
    


```python
bagging_predict_result
```




    [array([ 531170.32964683,  610437.80671003,  935793.13124286, ...,
             331964.44594363, 1009790.22112937,  451345.9132272 ]),
     array([495196.13568546, 646054.39267815, 986365.56528169, ...,
            331252.38335085, 907530.55639732, 468913.77372686]),
     array([512873.63822953, 654969.90479597, 998836.27669227, ...,
            327405.4302911 , 876943.02927024, 470934.42959936]),
     array([493825.84400474, 636737.39852681, 932594.32632075, ...,
            346914.54108486, 897040.92609115, 464065.61545808]),
     array([ 498631.77754482,  607710.21423254, 1020882.76294162, ...,
             335199.49778924,  936386.22238716,  424294.62431666]),
     array([506096.98682302, 669609.23719954, 940485.52597156, ...,
            345055.0638638 , 918021.10680715, 463670.6100106 ]),
     array([547657.77823348, 642343.41762371, 915100.66250346, ...,
            344643.67265125, 901723.62349688, 455516.83615459]),
     array([ 499624.58304481,  610091.26845662,  967500.51799746, ...,
             359938.28070706, 1006753.92094141,  465745.17602589]),
     array([525700.21446436, 653616.11499674, 997807.51664587, ...,
            346147.47470565, 979679.06092606, 478183.00379786]),
     array([504600.39478621, 651596.55745818, 952472.76765985, ...,
            334430.63067946, 905250.88060499, 443070.41117399])]




```python
# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(bagging_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    bagging_predict.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가
```


```python
# 예측한 결과값들의 평균을 계산하여 실제 테스트 데이트의 타겟변수와 비교하여 성능 평가

print("RMSE: {}".format(sqrt(mean_squared_error(bagging_predict, test_y)))) # RMSE
```

    RMSE: 210563.02109308538
    

