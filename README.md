## Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
## Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('XAUUSD_2010-2023.csv')

# Use the 'close' price column
close_prices = data['close'].dropna()

plt.rcParams['figure.figsize'] = [10, 7.5]

# Simulate ARMA(1,1) Process
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=len(close_prices))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plot_pacf(ARMA_1)

# Simulate ARMA(2,2) Process
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=len(close_prices) * 10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plot_pacf(ARMA_2)

```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img src="https://github.com/user-attachments/assets/809ab609-fa7d-4e6a-ada1-67795828501a" width="60%">

Partial Autocorrelation:

<img src="https://github.com/user-attachments/assets/392dd6f3-61d8-4b97-9091-1eae6218549f" width="60%">

Autocorrelation:

<img src="https://github.com/user-attachments/assets/dd4d8c15-9aad-4fea-a399-a7d2aea7d2a5" width="60%">

SIMULATED ARMA(2,2) PROCESS:

<img src="https://github.com/user-attachments/assets/c1529876-ce89-4bbc-b588-180518c4c077" width="60%">

Partial Autocorrelation:

<img src="https://github.com/user-attachments/assets/bd0d34be-51e0-423f-b3ce-c03618392602" width="60%">

Autocorrelation:

<img src="https://github.com/user-attachments/assets/23574890-5efb-4b31-b23c-94ab145151e5" width="60%">

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
