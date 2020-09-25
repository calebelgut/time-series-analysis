# *Using Time Series Data to Forecast Real Estate Value*
**Caleb Elgut - September 2020**

# Introduction

For this project I used a SARIMA model with time series data to forecast the next 10 years of ROI for 5 zip codes to help real estate investors know which zip codes were best to invest in. The two primary questions analyzed from this project are as follows:

1. What are the top 5 zip codes based on low urbanization, proximity to median value, level of risk, and ROI? 
1. After finding the above 5 zip codes, which will bring our investors the highest ROI over 10 years? 

The data provided came from the researchers at Zillow and consisted of 14,723 rows. The dataset included a small series of categorical features and a large number of columns, each of which contained the median value of homes in a zipcode for a certain month. The months ranged from April 1996 to April 2018.  

The following Python packages were used in the analysis of my data:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn
- Math
- Itertools
- StatsModels
- Pmdarima
- Warnings

# Who are the Stakeholders?

Real estate investors who are interested in rural property that will have a relatively low risk of investment with a relatively high ROI over the next 5-10 years. 

# What is Time Series Data? How do we analyze it?

Time series data refers to any dataset where the **progress of time** is an important dimension in the dataset. The data we are working with gives us *median real estate value over 12 years.* This will help us understand how the value of real estate in a given zip code changes over time. **Our job is to use this data to predict returns over the next few years.** Before we analyze our data with a model we must determine a few factors, particularly whether or not our data is stationary. If our data is not stationary, the time series model we use will not understand it and, therefore, will give us erroneous predictions.

## What do we mean when we call a time series data stationary?

A time series is *stationary* if its statistical properties such as mean, variance, etc. remain constant over time. Stationarity satisfies the constraints that the modeling requires. 

## How do we know if our data is stationary?

1. The mean of the series should not be a function of time. It should stay constant.
2. The variance of the series should not be a funciton of time.
  - This is a property known as *heteroscedasticity*
  - This means that the data should be gathered at the same interval.
3. The covariance of the *i*th term and the (*I + m*)th term should not be a function of time.
  - This means that the data should be gathered at the same frequency.

## Before we move further with time series, what does the initial dataset look like?

Before we get more into time series analysis & modeling we should work on answering this first question. In order to do that I would like to officially introduce the data to you. Here is what it initially looks like:

![Initial Dataset](/ReadMe4_Images/initial_dataset.png)

1. RegionID: ID
  - I initially was not sure if this column was a zip code or an ID but after a cursory search I discovered that 84654 is from Utah and not Chicago, IL
  RegionName: This was the column for zip codes.
  - 60657 is a Chicago ZipCode
2. City
3. State
4. Metro: What Metropolitan Area the zip code is nearest to.
5. CountyName: Name of the county the zip code resides within. 
6. SizeRank: Rank of Urbanization (The closer to 1, the more urbanized).
  - Our Firm is looking to invest in real estate that is not urbanized as their clients are looking to live in more rural environments.
  - **We will begin our EDA by creating a dataframe of the 25% least urbanized rows.**
7. All columns after the SizeRank correspond to median value of homes in the months between April, 1996 and April, 2018. Each column is dedicated to one month.

# Creating the list of our 5 top zip codes:

Our steps for attaining our top 5 zip codes were as follows:

## Examine the 25% least urban zip codes *by grabbing all rows corresponding to the bottom 25% of the SizeRank column.*

![bottom25zips](/ReadMe4_Images/bottom25zips.png)

Here we initially grab the top 75 zip codes based on Size Rank (urbanization level) and then create a dataframe that includes the zip codes and values for all zip codes with values higher than (aka: ranks lower than) the "top 75" zip codes

**We whiddled down our list of zip codes from 14,723 to 3,681.**

![bottom25zips2](/ReadMe4_Images/bottom25zips2.png)

## Next, grab all zip codes with values between 15% below and 15% above median,

Here we first create a new column called **yr_avg** that takes the mean of the past 12 months' median house value for each zip code. From here we look at the median value of this new column and grab all zip codes that exist between 15% below and 15% above the median value for this column. 

We grabbed these values because we want our clients to make a reasonable investment, none too high nor too low.

- Average Value 15% above median: $189875
- Average Value 15% below median: $124291

Our 3,681 zip codes were then whiddled down to 1,103.

## Now we want to add some descriptive features:
1. ROI: Profit/Investment: *Calculated by dividing the most recent value by the initial value and subtracting 1*
1. Standard Deviation of Monthly Values: *Calculated by using the NumPy method ".std" we look at the standard deviation from the initial value in 1996 to the most recent value in 2018*
1. Historical Mean Value: *Calculated by using the NumPy method ".mean" we look at the mean value from the initial value in 1996 to the most recent value in 2018*
1. The Coefficient of Variation (Measure of Relative Variability)
- This value will be key to understanding the unitary risk for our clients. It is the ratio of the standard deviation to the mean (average). 

![descrip_stats](/ReadMe4_Images/descrip_stats.png)

## Finally, we grab those zip codes with a maximum CV (volatile risk) of 0.6 and, within this risk profile, those that have the top 5 ROI:

![findtop5zips](/ReadMe4_Images/findtop5zips.jpg)

![top5zips](/ReadMe4_Images/top5zips.jpg)

### Our top 5 zip codes as per our search are:

1. 48894 - Westphalia, MI (ROI: 2.56, Population: 936 as of 2018)
1. 56360 - Osakis, MN (ROI: 2.29, Population: 1,749 as of 2018)
1. 40008 - Bloomfield, KY (ROI: 2.03, Population: 1,058 as of 2018)
1. 49339 - Pierson, MI (ROI: 1.9, Population: 171 as of 2018)
1. 27019 - Germanton, NC (ROI: 1.75, Population: 967 as of 2018)

It was good to see that each of our zip codes have ROIs of well over 100%! The rurality of the areas were confirmed by Googling their populations. **None of them broke 1,800. **

# Let the Preliminary Time Series Analysis Begin!

It is important to note that the above ranking of zip codes will change by the end of this project. While the ROIs I found above were based on historical data, the advice I am giving my stakeholders is for **the future.** 

It is here where the time series analysis truly begins! 

## Melt the Data! 

We begin by changing the format of our dataframe. If you recall, each month has its own separate column (for a total of over 260 columns!). This is not only inconvenient but it is not readable! Our model cannot process data like this. We needed a function to take the dataframe and put every month into one column and the values in a separate column. 

![melt_data](/ReadMe4_Images/melt_data.jpg)

From here, I melted down my data, set the time as the index (also important for the model to read), and then created a seperate dataframe for each zipcode with a monthly frequency. This means that when a time series model analyzes my data, it will giving me information on a *monthly* basis. 

![melt_in_action](/ReadMe4_Images/melt_in_action.jpg)

The resulting dataframe for the zip code 48894 looks like this: 

![48894df](/ReadMe4_Images/48894df.jpg)

## The Big Time Series 

After creating a list of dataframes, each containing dates and values for each of our 5 zip codes, I visualized a time series that reflected each zip code's change in value between 1996 and 2018. There was a clear lack of stationarity beginning with, as you can see, a clear upward trend.

![bigtimeseries](/ReadMe4_Images/bigtimeseries.png)

## Monthly Returns

Next, I wanted to look more closely at this data. My first decision was to look at the monthly change in value. Since a key element of time series analysis is looking at how the past affects the present I thought this would be a valuable piece of information. I added a column labeled "returns" that would reflect this and then visualized the monthly returns for each zipcode. 

As you can see in the code below, the monthly return is simply a month-to-month version of our ROI formula above. Instead of looking at the final date vs. the initial date, the monthly return looks at each month vs. the month before it. 

![monthly_returns_1](/ReadMe4_Images/monthly_returns_1.jpg)

The following is a visualization of the monthly returns of 48894.

![monthly_returns_2](/ReadMe4_Images/monthly_returns_2.png)

## Stationarity?

Remember earlier, we talked about stationarity? Here is where it matters! When we eventually run this data through a model that will then be used to forecast future ROIs for our zip codes, our data must have stationarity.

Stationarity is important for the same reason normal distributions are important: **We must satisfy the constraints that the model requires.**

If you look at the graph above, there seems to potentially be some seasonality although the trend we saw earlier doesn't look quite as prevalent.

### Examining the Rolling Mean & Standard Deviation

When we examine our data we can take what is called the rolling mean & standard deviation. What we do here is implement a **sliding window** that we place over our observations and plot their mean. **At any point in time *t* we can take the average/variance of the *m* last time periods. *m* is our window size. 

We want to see if our rolling mean changes much over time. If there is a great disparity, this can give us a hint that our data is not stationary and some actions must be taken to create stationarity. 

**To be clear, the returns over time for each zipcode are what we are measuring for stationarity**

In the below graphs you will see an example of a zipcode that turned out to be stationary (48894) and one that was not stationary (49339).

![stationary_graph](/ReadMe4_Images/stationary_graph.png)
![non_stationary_graph](/ReadMe4_Images/non_stationary_graph.png)

### Dickey-Fuller Test 

Regardless of what our eyes tell us, there is a test we can run to confirm stationarity: The Dickey-Fuller Test. 

This test comes from StatsModels. Its null hypothesis is that the time series is not stationary.

After running this test on each of the 5 zip codes it turned out that 3 had stationary data 2 did not. 

![dickeyfuller](/ReadMe4_Images/dickeyfuller.jpg)

### Differencing for Those Without Stationarity

For our two zip codes, 56360 and 49339, that did not have stationary data I decided to conduct differencing on their returns.

Differencing is a common method of dealing with both trend and seasonality in non-stationary time series data. In this technique, we take the *difference* of an observation at a particular instant of time with that at the previous instant. Here we took the difference of one year because data related to sales tend to have a yearly seasonality to them. 

After taking the difference of these two zip codes' returns, we achieve stationarity with them as well! **We can now instantiate time series for each individual zip code and move on to further analysis**

## Instantiating Time Series

Now that our zip codes are ready for analysis, we instantiate an individual time series for each zipcode by creating a label (TS_48894 for the zip code 48894, for example) and dropping the null values from the returns. 

For our two zip codes whose returns we differenced we create two time series: One with original non-stationary data and one with the stationary data that has been differenced. I denoted the differenced time series with a "d" as you can see below. The differenced time series will be used when examining the ACF and PACF. 

![timeseries](/ReadMe4_Images/timeseries.jpg)

## ACF, PACF, and Seasonal Plot Helper Functions

Here we have two functions to help us along the way, one that will examine the ACF and PACF over 5 years (this will be explained shortly) and another that uses the mean of the rolling data to examine seasonality. This will help us understand our p and q values for our SARIMA model (all to be explained shortly)

![helper_functions](/ReadMe4_Images/helper_functions.jpg)

# Let the Rest of the Time Series Analysis Begin!

## Using a GridSearch Function to Find Our Parameters 

The pmdarima library has a function called auto_arima which will take the zip code time series, an information criterion that we want to use to help us determine the best ARIMA parameters, the number of months, the number of days, the starting p-value, the starting q-value, the max for p & q, whether or not the search should be step-wise, and whether or not the search will be traced (recorded).

Let's explain some of these terms! 

1. **SARIMA**: The model we will be using. It stands for Seasonal Auto Regressive Integrated Moving Average
2. **Seasonal**: Used if there is seasonality detected in your data. 
3. **Auto Regressive**: Occurs when a value from a time series is regressed on previous values from the same time series. 
    - The order of this model is determined by analyzing the PACF and looking at its number of spikes. As we are using a Grid Search, we will not need to visually see this take place.
    - This model says that the value of today is based on the value of the previous day. (Or the value of this year is based on the value of the previous year)
4. **Moving Average**: The weighted sum of today's and yesterday's noise. (Noise = errors)
5. **Using ACF and PACF to understand the AR's p-value and the MA's q-value**: 
    - Re *ACF*: When looking at the sales at an individual point in time we must not only consider the most recent time period but previous time periods as well. Think of phone sales as an example. Whether or not someone buys an iPhone today will not only depend on whether or not they bought one yesterday, last month, or even last year but also on their purchasing habits in years prior. The same, of course, can be said of real estate purchases. The AutoCorrelation Function (ACF) looks at a specific day's purchase and relates it to previous time periods (in this case the periods are in years). The ACF is used to help choose the p-value for the AutoRegressive Model. 
      - **If the ACF shows us that the value of this year's data is only dependent on one previous year's data then we will use a p-value of 1 in the SARIMA Model. We will use a p-value of 2 if we see significant values in the months past month 24, etc. If we see no autocorrelation then the p-value will be equal to 0.** *Remember that we are using GridSearch so the function will determine this for us but we will also look at a plot of it as well.*
    - Re *PACF*: Instead of looking at the values, it looks at the error (or noise) in the previous value. When we conduct a Moving Average Model we focus on the PACF's number of spikes. This gives us an idea of how much impact yesterday's value has on the current value. 
      - AutoCorrelation doesn't give us individual impact but the overall impact, PACF gives us the individual impact.   
6. **The AIC Information Criterion**: This is how our grid search will determine the best possible model for us. We desire the model with the lowest AIC as this criterion will show us the model with the best goodness of fit *and* parsimony. Parsimony = simplicity. There may be models that will have a better goodness of fit than the model with the lowest AIC however the additional computational complexity it will require will not be worth the work. 
7. **p,d,q vs. P,D,Q, and s**: The lowercase letters p,d,q are the non-seasonal parameters. 
    - I described p & q already but I will also mention that the d is the **integrated** component of the ARIMA model. This value is concerned with the amount of *differencing* as it identifies the number of lag values to subtract from the current observation. *s* is the periodicity of the time series(ours will be set to 12 as we are looking at yearly periods). 
    - The uppercase letters are the exact same as the lowercase letters except these relate to the **seasonality** component of our time series. (P,D,Q are only used when the model is a SARIMA as opposed to an ARIMA)
        
## Our First Zip Code: 48894 - Westphalia, MI

To be clear, for the sake of time I will run through one zip code that had stationary data, one zip code whose data needed to be differenced, then I will present the best model of the series, and finally I will give a summary of all of the findings.

![acfpcf1](/ReadMe4_Images/acfpcf1.png)

Here we can see that the ACF tails off after around 12 lags. We can imagine that the d value will be 1. The PACF is difficult to grasp as the y-axis shows values as high as 25 in month 48. We will see what our gridsearch tells us. 

![gridsearch1](/ReadMe4_Images/gridsearch1.jpg)

For our non-seasonal order we have our pdq equal to (1,0,1) and for our seasonal order we have (0,0,2,12)

### Train-Test Split

From here I split the data into a training and testing set. The model is going to be trained on data preceding 2015 and the data will be tested on that which came after 2015. 

### Fit Model and Get Results

The SARIMA model was fit with our pdq, PDQ, and s. The results are below:

![sarima1](/ReadMe4_Images/sarima1.jpg)

The AR & MA terms with a lag of 1 both have very high correlations with very low p-values. We can see they are significant. The seasonal MA value after 12 months has a moderately strong negative correlation and also a low p-value. The seasonal MA value after a lag of 24 months, however, has a very high p-value. We can disregard it. 

The error-analysis is below:

![error1](/ReadMe4_Images/error1.png)

Here we can see that the model did not capture all of the data's signal. In the top-left graph we can see that our standardized residuals are not quite white-noise as there is variance in the mean after 2010. In the top right graph we should have a KDE of around a normal-distribution. We do not have that. Finally, our errors should be distributed linearly and if we look at the bottom left graph we do not have this. 

There is more work that needs to be done on this model. I trained the model on data from other zip codes and as I fit it with more data, the error results became more along the lines of what we want. 

After analyzing the RMSE of our train and test data we can see that the predicted values are not too far off from our actual values. The RMSEs are similar, as well, so it seems like this model is fit fairly well even given our error issues above. 

![traindata1](/ReadMe4_Images/traindata1.png)
![testdata1](/ReadMe4_Images/testdata1.png)

### Forecast

Below is the result of forecasting from this model:

![forecast1](/ReadMe4_Images/forecast1.png)

While the returns do not look like they will change too much over the next 10 years, after analyzing the predicted mean return for years 1, 3, 5, and 10 we received the following information:
1. Total expected return in 1 year: -4.1%
1. Total expected return in 3 years: -7.34%
1. Total expected return in 5 year: -7.99%
1. Total expected return in 10 years: -8.18%

# Results after repeated measures:

This process above was repeated for the other 4 zip codes to determine which zip codes would have the best ROI over the next 5-10 years. Below you will see the best model I was able to acquire as well as the list of our suggested zip codes.

## Best Model: Zip Code 40008

- Order: (0,1,4) ; Seasonal Order: (0,0,0)[0]

SARIMAX RESULTS:

![sarima2](/ReadMe4_Images/sarima2.jpg)

Error Results:

![error2](/ReadMe4_Images/error2.png)

- Train RMSE: 0.00374, Test RMSE: 0.00480

![traindata2](/ReadMe4_Images/traindata2.png)
![testdata2](/ReadMe4_Images/testdata2.png)

Forecast:

1. Total expected return in 1 year: 27.87%
1. Total expected return in 3 years: 107.3%
1. Total expected return in 5 year: 236.09%
1. Total expected return in 10 years: 1047.71%

# Final Results and Recommendations:

After we performed a time series analysis on our 5 zip codes and created a dataframe that orders the returns through 10 years, the best zipcodes to invest in would be:

1. Bloomfield, KY (40008) 10 Year ROI of 10.47 (1,047%)
1. Osakis, MN (56360) - 10 Year ROI of 5.75 (575%)
1. Pierson, MI (49339) - 10 Year ROI of 1.82 (182%)

The ROI DataFrame is Below: 

![final_roi](/ReadMe4_Images/final_roi.jpg)
