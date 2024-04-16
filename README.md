# Metro-Interstate-Traffic-Volume-Prediction

Utilized Pandas, Matplotlib, Seaborn, NumPy, and Scikit-learn for comprehensive data analysis, visualization, preprocessing, and predictive modeling to forecast traffic volume.

## Data

Data directory contains a train set and a test set we were given by [123ofAI](https://www.123ofai.com) during our hackathon. If you want the raw data, check out here: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

## Project Methodology

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/ML%20Workflow.jpg)


### Exploratory Data Analysis (EDA):

- Reading the training data from a CSV file.
- Displaying the first few records (df.head()).
- Checking dataset information (df.info()) including field types, non-null values, and data types.
- Analyzing categorical variables like 'holiday', 'weather_main', and 'weather_description'.
- Visualizing data distributions and relationships using histograms, box plots, and count plots.

### Data Preprocessing:

- Handling missing (NULL) values by dropping or imputing.
- Removing duplicate rows.
- Dropping irrelevant features ('snow_1h', 'rain_1h', 'weather_description').
- Handling categorical variables by one-hot encoding ('holiday', 'weather_main').

### Feature Engineering:

- Extracting date-time features like year, month, weekday, and hour from 'date_time'.
- Creating a new feature 'day_part' based on the hour of the day.
- One-hot encoding the new categorical feature 'day_part'.

### Correlation Analysis:

- Calculating and visualizing the correlation matrix between numerical features and the target variable ('traffic_volume') before and after feature engineering is done on the data.

### Feature Importance and Selection:

- Using Random Forest Regressor to determine feature importance.
- Selecting the top features based on importance scores.

### Model Development:

- Splitting the dataset into training and validation sets.
- Scaling the features using StandardScaler.
- Experimenting with different regression models including Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression, SVR, XGB Regressor, Hist Gradient Boost Regressor, Random Forest Regressor, K Neighbors Regressor, and Decision Tree Regressor.
- Tuning hyperparameters using GridSearchCV.
- Training models, evaluating their performance using metrics like R-squared score and Mean Squared Error, and selecting the best model.

### Testing and Creating Output CSV:

- Reading the test data from a CSV file.
- Preprocessing the test data similar to training data.
- Scaling the test data using the same scaler used for training data.
- Making predictions using the trained model (Hist Gradient Boost Regressor).
- Creating a submission CSV file containing predictions.

## Visualizations and their interpretation

### Dataset statistics
- We find that each feature has apporoximately 10% of Null values

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/data_info.png)


- 'rain_1h' and 'snow_1h' features are represented poorly in the Dataset.

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/data_describe.png)


- We have three Multi Categorical Variables 'holiday', 'weather_main', 'weather_description'

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/categorical_data_describe.png)


**Inference From the above Dataset**

- Each feature has apporoximately 10% of Null values except the 'holiday' feature.
- In the holidays feature, we have only 42 rows (1%) where a holiday is mentioned. Rest are simply Null values.
- ‘weather_main’ feature effectively summarizes the weather data.
- Rain and Snow features being seasonal, majority of data have 0 values, and hence would not be relevant for the model


### Univariate analysis

- Now, we'll explore the data distribution of the numerical features and the target variable as well
![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/hist_1.png)
![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/hist_2.png)



**We can clearly see from the above graphs, there is poor representation of rain_1h and snow_1h in the dataset.**

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/clouds%20all%20box.png)


**The distribution of 'traffic volume' appears highly erratic in relation to the 'clouds_all' feature.**

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/temp%20box.png)

**Temperature feature has an anamoly around 0°C**





- Next we will explore the Multi Categorical features 'holiday', 'weather description' and 'weather_main'.


![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/holiday_bar.png)


**Bar chart removing the rows ‘None’ to have a look just of the holidays**

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/weather%20main%20count.png)


**Visualization of the 'weather_main' category distribution in the dataset`**


![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/weather%20description%20bar.png)


**Visualization of the 'weather_description' category distribution in the dataset`**




### Bivariate analysis

- Holiday feature vs Traffic Volume
![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/holiday%20vs%20traffic%20volume.png)


**The distribution of the traffic volume during the public holidays has on average low values. There is an exception: “New Years Day” is a holiday that reaches very high traffic volume**

- Weather Main feature vs Traffic Volume
![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/weather%20main%20vs%20traffic%20volume.png)


**Among the various types of weather, the ones that have on average lower traffic volumes are ‘Squall’ and ‘Fog’. The weather for which are registered, on average, major traffic volumes are ‘Clouds’ and ‘Haze’.**

- Clouds All feature vs Traffic Volume
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/clouds_all_vs_traffic_volume_graph.png)


**The distribution of 'traffic volume' appears highly erratic in relation to the 'clouds_all' feature.**




### Correlation between the features
 - Correlation tests are often used in feature selection, where the goal is to identify the most relevant features (variables) for a predictive model. Features with high correlation with the target variable are often considered important for prediction.
 - While correlation analysis is useful for identifying relationships between variables, it is important to note that correlation does not necessarily imply causation. Simply because two factors vary together based on the available data does not mean that one factor causes changes in the other. There could be some third, underlying variable influencing both.

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/correlation%20before%20feature%20engineering.png)


**As we can see, there is no strong correlation between the features**

## Data Cleaning and Pre Processing
- **Pre-processing (Cleaning): Address missing (NULL) values - drop or imputation.**
    - **We will use the ffill() method**
    ```
    data.ffill(inplace = True)
    ```
	
- **Since we have already seen poor reperesentation of 'snow_1h' and 'rain_1h', we will drop the these features for the model.**
- **I decided to remove the 'weather_description' feature because the information I could get from it was not strictly necessary, given the feature ‘weather_main’ which effectively summarizes the weather data.**
    ```
    updated_data  = data.drop(['snow_1h', 'rain_1h','weather_description'] , axis =1)
    ```
- **Converting 'holiday' feature into just 'holiday' and 'Work day'.**
    ```
    updated_data['holiday'] = updated_data['holiday'].apply(lambda x: 'Work day' if pd.isna(x) else 'holiday' ) 
    ```
- **Next we will first convert the 'date_time' feature into a pandas datetime object.**
    ```
    updated_data['date_time'] = pd.to_datetime(updated_data['date_time'], format = '%d-%m-%Y %H:%M')
    ```
- **We now extract the 'year', 'month', 'weekday' and 'hour' from the datetime object.**
    ```
    updated_data['year'] = updated_data['date_time'].dt.year
    updated_data['month'] = updated_data['date_time'].dt.month
    updated_data['weekday'] = updated_data['date_time'].dt.weekday
    updated_data['hours'] = updated_data['date_time'].dt.hour
    ```
- **Next we will now divide the 24 hours of the day into 'before_sunrise', 'after_sunrise', 'afternoon' and 'night' categories.**
    ```
    updated_data['hours'].unique()
    ```
- **We will create a function ,which will split the hours into the above four categories.**
    ```
    def day_part(hour):
        day_part = ''
        if hour in [1,2,3,4,5]:
            day_part = 'before_sunrise'
        elif hour in [6,7,8,9,10,11,12]:
            day_part = 'after_sunrise'
        elif hour in [13,14, 15, 16, 17, 18]:
            day_part = 'evening'
        else :
            day_part = 'night'
        return day_part
    ```
- **Using the map() function to loop through the 'hours' feature and based on the hour - value we will allot the 4 day-sections. This way we will create one more feature 'day_part' in our existing dataset.**
    ```
    updated_data['day_part'] = updated_data['hours'].map(day_part)
    ```
- **Next we use the pd.get_dummies function to one hot encode the categorical features 'holiday', 'weather_main' and 'day_part'.**
    ```
    updated_data = pd.get_dummies(data1, columns =['holiday', 'weather_main','day_part'])
    ```
- **Finally we index our dataset by 'date_time'**
    ```
    updated_data.set_index('date_time',inplace = True)
    ```
### Correlation testing (After feature engineering)

    ```
    corr_updated_data  = updated_data.corr()
    fig, ax = plt.subplots(figsize = (15, 10))
    plt.xticks(rotation =45)
    sns.heatmap(corr_updated_data, annot = True, linewidths = .5, fmt = '.1f', ax = ax)
    plt.show()
    ```
![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/correlation%20after%20feature%20engineering.png)


**We can see that there is a negative correlation between day_part_before_sunrise and traffic_volume and hour features**

## Feature Importance and Selection Using Random Forest Regressor

Feature importance and selection with the Random Forest Regressor involve identifying the most influential features in predicting the target variable.

**Feature Importance:** Random Forest Regressor calculates feature importance based on how much the tree nodes that use that feature reduce impurity across all trees in the forest. Features that lead to large reductions in impurity when used at the root of a decision tree are considered more important. Random Forest assigns a score to each feature, indicating its importance. Higher scores signify more important features.

**Visualizing Feature Importance:** Plotting the feature importance scores can provide insights into which features are most relevant for prediction. This visualization can aid in understanding the data and making decisions about feature selection.

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/important%20features.png)


In summary, feature importance and selection with Random Forest Regressor involve identifying and prioritizing features based on their contribution to predicting the target variable. This process can enhance model performance, interpretability, and understanding of the underlying data.

## Model Development

- **we will select just the top 7 features that we got from the Random Forest Regressor**
    ```
    important_features = [ 'hour','temp','weekday','day_section_night','month', 'year','clouds_all']
    ```
- Splitting the dataset into training and **validation set**. This validation set is to assess the performance of the model during training
- For the hackathon we have our test dataset without the ground truth values
     

- **Scaling : we do the scaling of the data using the StandardScaler() function from sklearn**

- **Experimenting with different models and hyperparameters(using GridSaerchCV), so that we can select the best model for our submission**

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/best%20model.png)
![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/hyperparameters.png)


- **Selecting the best model**
    ```
    hgbr = HistGradientBoostingRegressor(max_iter=1000, random_state=32)
    hgbr.fit(x_train_scaled, y_train)
    y_pred = hgbr.predict(x_test_scaled)
    print(f"r2 score : {r2_score(y_test, y_pred)} mean squared error : {mean_squared_error(y_test, y_pred)} mean absolute error : {mean_absolute_error(y_test,y_pred)} ")
    ```




## Key takeaways

- I understood what a typical ML workflow (model development) might look like in a real-world scenario.
- Majority of the algorithms used are still a black box for me, curious and looking forward to learn what goes in the backend of these algorithms.
- The importance of thorough **data preprocessing** cannot be overstated, as it significantly impacts model performance.
- **Feature engineering** plays a crucial role in improving model accuracy by extracting relevant information from raw data.
- Understanding **feature correlations** helps in identifying redundant features and refining the feature set.
- **Model selection and tuning** are iterative processes requiring experimentation and evaluation of various algorithms and hyperparameters.
- Effective **visualization** enhances data exploration and aids in gaining actionable insights for model improvement.

## Acknowledgements

 - [Metro Interstate Traffic Prediction](https://www.kaggle.com/code/meemr5/traffic-volume-prediction-time-series-starter/notebook)
 - [Feature importance using Random forest regressor](https://mljar.com/blog/feature-importance-in-random-forest/)
 - [Time Series Analysis](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at karanshah51101@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

