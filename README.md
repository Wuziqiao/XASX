# XASX-Model

This is the implementions for the paper work of "Highly-Accurate Electricity Load Forecasting via Knowledge Aggregation"
## Brief Introduction

 This paper proposes a novel hybrid LF model, named XASX. Its main idea is two-fold: 1) it exploits domain knowledge to decompose the raw load series in-to three independent components of trendiness, seasonality, and irregularity, and 2) individualized ML models are established to capture the sequential patterns of each decomposed component.Four real-world datasets gathering the historical power consumption of Chinese cities are adopted to benchmark the empirical studies. The results substantiate that our model outperforms nine state-of-the-art LF competitors, including both classic time-series and ML models in terms of higher accuracy and less forecasting bias. 
## Enviroment Requirement

The Code has been tested running under Python 3.7.0
###requirement

numpy

scipy

sklearn

pandas

torch
## Dataset

We select the monthly electricity consumption of four Chinese cities as the benchmark datasets, named D1, D2, D3, D4, respectively. They have different time ranges—from January 2013 to De-cember 2021 (D1, D2, D3) and from January 2016 to December 2021 (D4). Besides, the four datasets also include corresponding features of holiday information and weather information of average temperature, humidity, wind, rainfall, air pressure, cloud cover, maximum temperature, and minimum temperature.

## Parameters setting
1)	Learning rate in XGBoost: lr;
2)	Max_depth in XGBoost:max_depth ;
3)	Number of estimators in XGBoost: n_esm;
4)	Number of autoregressive terms: p;
5)	Moving average number of terms: q;








