# Electricity Transformer Dataset (ETDataset) 

In this Github repo, we provide several datasets could be used for the long sequence time-series problem. All datasets have been preprocessed and they were stored as `.csv` files.  The dataset ranges from 2016/07 to 2018/07.

**Dataset list** (updating)

- [x] **ETT-small**: The data of 2 Electricity Transformers at 2 stations, including load, oil temperature.
- [x] **Cryptocurrency**: The data of 14 diffrent cryptocurrency, including Open, Close, highest, and lowest price every 1minas well as the change of the price .


## ETT-small:

We donated two years of data, in which each data point is recorded every minute (marked by *m*), and they were from two regions of a province of China, named ETT-small-m1 and ETT-small-m2, respectively. Each dataset contains 2 year * 365 days * 24 hours * 4 times = 70,080 data point. Besides, we also provide the hourly-level variants for fast development (marked by *h*), i.e. ETT-small-h1 and ETT-small-h2. Each data point consists of 8 features, including the date of the point, the predictive value "oil temperature", and 6 different types of external power load features. 

<p align="center">
<img src="./img/appendix_dataset_year.png" height = "200" alt="" align=center />
<img src="./img/appendix_auto_correlation.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b>The overall view of "OT" in the ETT-small.&nbsp;&nbsp;&nbsp;&nbsp;<b>Figure 2.</b>The autocorrelation graph of all variables.
</p>

Specifically, the dataset combines short-term periodical patterns, long-term periodical patterns, long-term trends, and many irregular patterns. We firstly give an overall view in Figure 1, and it shows evident seasonal trends. To better examine the existence of long-term and short-term repetitive patterns, we plot the autorcorrelation graph for all the variables of the ETT-small-h1 dataset in Figure 2. The blue line in the above is the target 'oil temperature', and it maintains some short-term local continuity. However, the other variables (power load) shows short-term daily pattern (every 24 hours) and long-term week pattern (every 7 days).

We use the `.csv` file format to save the data, a demo of the ETT-small data is illustrated in Figure 3. The first line (8 columns) is the horizontal header and includes "date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL" and "OT". The detailed meaning of each column name is shown in the Table 1.


| Field | date | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Description | The recorded **date** |**H**igh **U**se**F**ul **L**oad | **H**igh **U**se**L**ess **L**oad | **M**iddle **U**se**F**ul **L**oad | **M**iddle **U**se**L**ess **L**oad | **L**ow **U**se**F**ul **L**oad | **L**ow **U**se**L**ess **L**oad | **O**il **T**emperature (target) |

<p align="center"><b>Table 1.</b> Description for each columm.</p>

## Cryptocurrency:

![image](./Images/crypto_cube.png)

### **train**
This dataset contains historical trading data for multiple cryptocurrencies such as Bitcoin and Ethereum. Timestamps are represented in minutes and correspond to ID numbers which map to specific cryptocurrencies. Columns include the total number of trades, open, high, low and close prices, trade volume, volume weighted average price, and target residual log-returns over a 15 minute period.

| Column | Description |
|---|---|
| `timestamp` | All timestamps are returned as second Unix timestamps (the number of seconds elapsed since 1970-01-01 00:00:00.000 UTC). Timestamps in this dataset are multiple of 60, indicating minute-by-minute data. |
| `Asset_ID` | This column contains an ID number that corresponds to a specific cryptocurrency. For example, an Asset_ID of 1 corresponds to Bitcoin. A mapping from Asset_ID to the name of the cryptocurrency can be found in a separate file called `asset_details.csv`. |
| `Count` | Total number of trades in the time interval (last minute). |
| `Open` | Opening price of the time interval (in USD). |
| `High` | Highest price reached during time interval (in USD). |
| `Low` | Lowest price reached during time interval (in USD). |
| `Close` | Closing price of the time interval (in USD). |
| `Volume` | The quantity of the cryptocurrency that was bought or sold during the minute of trading activity represented by the corresponding timestamp, measured in USD. |
| `VWAP` | The average price of the asset over the time interval, weighted by volume. VWAP is an aggregated form of trade data. |
| `Target` | This column contains the residual log-returns for the cryptocurrency over a 15-minute horizon. Residual log-returns are a way of measuring the percentage change in the price of an asset over a given time period, relative to some baseline. The Target column in this dataset represents the residual log-returns for a 15-minute period starting from the minute represented by the corresponding timestamp. |

### **asset_details**

Additional information about cryptocurrencies includes:

| Column | Description |
|---|---|
| `Asset_ID` | An ID code for the cryptocurrency. |
| `Asset_Name` | The real name of the cryptocurrency associated to Asset_ID. |
| `Weight` | The weight that the cryptocurrency associated to Asset_ID receives in the evaluation metric. |


