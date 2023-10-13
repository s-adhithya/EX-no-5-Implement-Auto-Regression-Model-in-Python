# EX-no-5-Implement-Auto-Regression-Model-in-Python

## AIM:
To Implement Auto Regression Model in Python

## ALGORITHM:

1. Import the necessary packages

2. Plot the Dataset using matplotlib

3. Implement the adfuller and plot_pacf,plot_acf

4. Train your model and taking test value from dataset

5. Represent the result in graphical representation as given below.

## PROGRAM:
```
!pip install statsmodels --upgrade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
df = pd.read_csv('/content/rainfall.csv', index_col=0, parse_dates=True)
df.shape
df.pop("preciptype")
df.pop("dew")
df.pop("humidity")
df.pop("sealevelpressure")
df.pop("winddir")
df.pop("solarradiation")
df.pop("windspeed")
df.pop("precipprob")
df.head()
x=df.values
df.plot()
from statsmodels.tsa.stattools import adfuller
dftest= adfuller(df['temp'],autolag='AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ",dftest[1])
print("3. Number Of Lags : ",dftest[2])
print("4.Num of observation used FOr ADF Regression  and Critical value Calculation :",dftest[3])
for key,val in dftest[4].items():
     print("\t",key, ":",val)
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(df['temp'],lags=25)
acf=plot_acf(df['temp'],lags=25)
train=x[:len(x)-600]
test=x[len(x)-600:]
model=AutoReg(train,lags=40).fit()
print(model.summary())
pred=model.predict(start=len(train),end=len(x)-1,dynamic=False)
from matplotlib import pyplot
pyplot.plot(pred)
pyplot.plot(test,color='green')
print(pred)
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=sqrt(mean_squared_error(test,pred))
rmse
pred_future=model.predict(start=len(x)+1,end=len(x)+7,dynamic=False)
print("future prediction")
print(pred_future)
```
## OUTPUT:
### df.shape
![Screenshot 2023-10-13 154957](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/23c7a646-2e5f-4237-8e40-dfc219f71a82)


### df.head()
![Screenshot 2023-10-13 155019](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/41352e67-68e1-4f0e-b1ca-c73273bdff43)

### df.plot()
![Screenshot 2023-10-13 155044](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/24d37d16-0315-4902-a13b-338de1579684)

### adfuller
![Screenshot 2023-10-13 155101](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/32d17a28-c365-413e-adcb-80de5f097011)


### Partial Autocorrelation
![Screenshot 2023-10-13 155146](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/1914b305-a1ca-41a3-8339-6f96ff20d701)


### Autocorrelation
![Screenshot 2023-10-13 155156](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/ace6fd9b-bf16-4c43-9e47-e09f59064798)


### Autoreg Model Result
![Screenshot 2023-10-13 155216](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/d67c2ace-eb1c-44ca-8be9-93a024ecefe6)


### Predicted Values and Plot

![Screenshot 2023-10-13 155307](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/fdea938f-b1f6-47fd-a2f2-61087bc36b4c)


### RMSE Value
![Screenshot 2023-10-13 155216](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/d67c2ace-eb1c-44ca-8be9-93a024ecefe6)


### Future Prediction
![Screenshot 2023-10-13 155330](https://github.com/s-adhithya/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/113497423/0203df1a-379d-42c6-8e32-03fa01975168)


## RESULT:
Thus we have successfully implemented the Auto Regression Model using Python program.

