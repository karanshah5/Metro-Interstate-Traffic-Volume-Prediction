![Traffic Banner](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/TrafficBanner.jpg)

# Project Title

Metro Interstate Traffic Volume Prediction

## Implementation Details

- Dataset: Metro Interstate Traffic Volume Dataset (view below for more details)
- Model: [HistGradientBoostingRegressor]('https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html')
- Input: 8 features - Holiday, Temp, Weather Description ...
- Output: Traffic Volume.

## Dataset Details

[This dataset was obtained from this repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

Metro Interstate Traffic Volume  dataset is a collection of traffic volume data observed on a section of interstate highway in the Minneapolis-St Paul metropolitan area in Minnesota, USA. This dataset includes hourly traffic volume measurements along with corresponding attributes such as date, time, weather conditions, and holiday indicators. The data spans from 2012 to 2018, providing a comprehensive view of traffic patterns over several years. This dataset is valuable for studying and predicting traffic volume fluctuations based on various factors, making it suitable for machine learning tasks such as regression and time series analysis.

### Variables Table of the above dataset

 | Variable Name      | Role    | Type       | Description       | Unit       |
|--------------------|---------|------------|-------------------|------------|
| holiday            | Feature | Categorical| US National holidays plus regional holiday, Minnesota State Fair | -         |
| temp               | Feature | Continuous | Average temp in kelvin | Kelvin    |
| rain_1h            | Feature | Continuous | Amount in mm of rain that occurred in the hour | mm         |
| snow_1h            | Feature | Continuous | Amount in mm of snow that occurred in the hour | mm         |
| clouds_all         | Feature | Integer    | Percentage of cloud cover | %          |
| weather_main       | Feature | Categorical| Short textual description of the current weather | -          |
| weather_description| Feature | Categorical| Longer textual description of the current weather | -          |
| date_time          | Feature | Date       | Hour of the data collected in local CST time | -          |
| traffic_volume     | Target  | Integer    | Hourly I-94 ATR 301 reported westbound traffic volume | -          |


## Evaluation and Results

### Exploring the dataset statsitics

- We find from the evaluation of the dataset that all the features have approximately 10% null values.

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/data_info.png)

- 'rain_1h' and 'snow_1h' features are poorly represented in the Dataset.

![alt text](https://github.com/karanshah5/Metro-Interstate-Traffic-Volume-Prediction/blob/main/images/data_describe.png)

- We have three Multi Categorical Variables 'holiday', 'weather_main', 'weather_description.

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

**It's evident from the graphs that 'rain_1h' and 'snow_1h' have inadequate representation in the dataset.**

- Next we will explore the Categorical features 'holiday', and 'weather_main'.


![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/holiday_graph.png)
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/trafficVol_temp_univariate.png)

### Visualization - Bivariate analysis

- Clouds All feature vs Traffic Volume
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/clouds_all_vs_traffic_volume_graph.png)

- Weather Main feature vs Traffic Volume
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/weather_main_vs_traffic_volume_graph.png)



### Correlation between the features
  Correlation tests are often used in feature selection, where the goal is to identify the most relevant features (variables) for a predictive model. Features with high correlation with the target variable are often considered important for prediction. However, it's essential to note that correlation does not imply causation, and other factors such as domain knowledge and data quality should also be considered in feature selection.

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/correlation_graph_initial_dataset.png)

**As we can see, there is no strong correlation between the features**

## Data Cleaning and Pre Processing
- **Pre-processing (Cleaning): Address missing (NULL) values - drop or imputation.**
    - **we will use the ffill() method**
    ```
    data.ffill(inplace = True)
    ```
	
- **Since we have already seen poor reperesentation of 'snow_1h' and 'rain_1h', and similarity between weather_main and  'weather_description' we will drop the three features for the model.**
    ```
    data1 = data.drop(['snow_1h', 'rain_1h','weather_description'] , axis =1)
    ```
- **Converting 'holiday' feature into just holiday and 'unknown'.**
    ```
    data1['holiday'] = data1['holiday'].apply(lambda x: 'unknown' if pd.isna(x) else 'holiday' ) 
    ```
- **Next we will first convert the 'date_time' feature into a pandas datetime object.**
    ```
    data1['date_time'] = pd.to_datetime(data1['date_time'], format = '%d-%m-%Y %H:%M')
    ```
- **We now extract the 'year', 'month', 'weekday' and 'hour' from the datetime object.**
    ```
    data1['year'] = data1['date_time'].dt.year
    data1['month'] = data1['date_time'].dt.month
    data1['weekday'] = data1['date_time'].dt.weekday
    data1['hour'] = data1['date_time'].dt.hour
    ```
- **Next we will now divide the 24 hours of the day into 'before_sunrise', 'after_sunrise', 'afternoon' and 'night' categories.**
    ```
    data1['hour'].unique()
    ```
- **We will create a function ,which will split the hours into the above four categories.**
    ```
    def day_category(hour):
        day_section = ''
        if hour in [1,2,3,4,5]:
            day_section = 'before_sunrise'
        elif hour in [6,7,8,9,10,11,12]:
            day_section = 'after_sunrise'
        elif hour in [13,14, 15, 16, 17, 18]:
            day_section = 'evening'
        else :
            day_section = 'night'
        return day_section
    ```
- **Using the map() function to loop through the 'hour' feature and based on the hour - value we will allot the 4 day-sections. This way we will create one more feature 'day_section' in our existing dataset.**
    ```
    data1['day_section'] = data1['hour'].map(day_category)
    ```
- **Next we use the pd.get_dummies function to do one hot encoding of the categorical features 'holiday', 'weather_main' and 'day section'.**
    ```
    data1 = pd.get_dummies(data1, columns =['holiday', 'weather_main','day_section'])
    ```
- **Finally we set the feature 'date_time' as row index in our dataset.**
    ```
    data1.set_index('date_time',inplace = True)
    ```
### Correlation testing - second time
- **After the above feature engineering.**

    ```
    corr_data1 = data1.corr()
    fig, ax = plt.subplots(figsize = (15, 10))
    plt.xticks(rotation =45)
    sns.heatmap(corr_data1, annot = True, linewidths = .5, fmt = '.1f', ax = ax)
    plt.show()
    ```
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/correlation_graph_after_feature_engineering.png)

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/caption.png)

## Feature Importance and Selection Using Random Forest Regressor

Feature importance and selection with the Random Forest Regressor involve identifying the most influential features in predicting the target variable.

**Feature Importance:** Random Forest Regressor calculates feature importance based on how much the tree nodes that use that feature reduce impurity across all trees in the forest. Features that lead to large reductions in impurity when used at the root of a decision tree are considered more important. Random Forest assigns a score to each feature, indicating its importance. Higher scores signify more important features.

**Visualizing Feature Importance:** Plotting the feature importance scores can provide insights into which features are most relevant for prediction. This visualization can aid in understanding the data and making decisions about feature selection.

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/feature_selection.png)

In summary, feature importance and selection with Random Forest Regressor involve identifying and prioritizing features based on their contribution to predicting the target variable. This process can enhance model performance, interpretability, and understanding of the underlying data.

## Model Development

- **we will select just the top 7 features that we got from the Random Forest Regressor**
    ```
    important_features = [ 'hour','temp','weekday','day_section_night','month', 'year','clouds_all']
    ```
- Splitting the dataset into training and **validation set**. This validation set is to test our model internally before submitting it to the test set
    - *Note : we have already been provided the test data set for the hackathon*

- **Scaling : we do the scaling of the data using the StandardScaler() function from sklearn**

- **Experimenting with different models , so that we can select the best model for our submision**

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/experimenting_models.png)

- **Selecting the best model**
    ```
    regrh = HistGradientBoostingRegressor(random_state=32)
    regrh.fit(x_train_scaled, y_train)
    y_pred = regrh.predict(x_test_scaled)
    print(f"r2 score : {r2_score(y_test, y_pred)} \n mean squared error : {mean_squared_error(y_test, y_pred)} \n mean absolute error : {mean_absolute_error(y_test,y_pred)} ")
    ```

## Testing and Creating Output CSV

- **we repeat the same process of data cleaing, pre processing, scaling etc with the test data.**
- **finally we submit the submission file.**


## Key Takeaways

After the hackathon process, key takeaways include:

1. **Data Exploration is Crucial**: Understanding the dataset thoroughly is essential before building any machine learning model. Exploratory data analysis helps in identifying patterns, outliers, and relationships within the data.

2. **Feature Engineering Matters**: Creating meaningful features from the existing data can significantly improve model performance. Techniques like encoding categorical variables, creating new features from datetime data, and scaling numerical features might be beneficial.

3. **Model Selection and Tuning**: Experimenting with various machine learning algorithms and hyperparameters can lead to improved performance. Techniques like cross-validation and hyperparameter tuning help in selecting the best model configuration.

4. **Evaluation Metrics**: Choosing the right evaluation metric based on the problem domain is crucial. In regression tasks like traffic volume prediction, metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) are commonly used to evaluate model performance.

5. **Interpreting Model Results**: Understanding how the model makes predictions and the importance of different features can provide valuable insights into the problem domain. Techniques like feature importance analysis help in understanding which features contribute most to the model's predictions.

6. **Continuous Learning**: Hackathons are great learning experiences, and reflecting on what worked well and what could be improved prepares you for future challenges. Continuous learning and experimentation are key to mastering machine learning techniques.

Overall, participating in hackathons provides valuable hands-on experience in solving real-world problems using machine learning techniques. 


## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

The next steps would be 

- Incorporate chosen features into model development.
- Train the model and assess its performance through rigorous evaluation.
- Fine-tune the model if necessary for optimization.
- Analyze model predictions for insights into the problem domain.
- Deploy the model and monitor its performance, iterating as needed for continuous improvement.


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

### 1) How does the HistGradientBoostingRegressor model work ?

[HistGradientBoostingRegressor is a gradient boosting machine learning algorithm introduced in scikit-learn.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

1. **Histogram-based Approach**: Unlike traditional gradient boosting methods, HistGradientBoostingRegressor operates on histograms of the features, which improves computational efficiency, especially for large datasets.

2. **Gradient Boosting Technique**: It builds an ensemble of decision trees sequentially, where each tree corrects the errors made by the previous trees. It minimizes a loss function, typically mean squared error (MSE) for regression tasks, by iteratively fitting new trees to the residuals of the previous predictions.

3. **Gradient and Hessian Computation**: It approximates the gradient and the Hessian of the loss function for each histogram bin, which allows for efficient computation of the split points during tree construction.

4. **Regularization and Shrinkage**: HistGradientBoostingRegressor incorporates regularization techniques like shrinkage and tree pruning to prevent overfitting and improve generalization performance.

5. **Quantile Estimation**: It supports quantile regression, allowing the estimation of conditional quantiles of the target variable, which is useful for uncertainty estimation and prediction intervals.

6. **Parallelism and Efficiency**: It utilizes multi-threading and other optimizations to efficiently handle large datasets and improve training speed.

Overall, HistGradientBoostingRegressor combines the benefits of histogram-based techniques with the power of gradient boosting, resulting in a highly efficient and scalable algorithm for regression tasks.

### 2) How do you train the model on a new dataset?

To train the `HistGradientBoostingRegressor` model on a new dataset, you need to follow these general steps:

1. **Prepare the Data**: Ensure your new dataset is properly preprocessed and formatted. This includes handling missing values, encoding categorical variables, and splitting the data into features (X) and target variable (y).

2. **Import the Necessary Libraries**: Import the required libraries, including `HistGradientBoostingRegressor` from scikit-learn and any other libraries needed for data preprocessing and evaluation.

3. **Instantiate the Model**: Create an instance of the `HistGradientBoostingRegressor` model. You can optionally specify hyperparameters during instantiation, or you can use the default settings.

4. **Fit the Model to the Data**: Use the `fit()` method of the model to train it on your new dataset. Pass the features (X) and the corresponding target variable (y) to this method.

5. **Evaluate Model Performance (Optional)**: After training, evaluate the performance of the model on a separate validation dataset (if available). Use appropriate evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or others to assess how well the model generalizes to new data.

6. **Make Predictions (Optional)**: Once trained, you can use the trained model to make predictions on new, unseen data. Use the `predict()` method and pass the new features as input to get the predicted target variable values.


Adjust the code according to your specific dataset and requirements. Ensure proper data preprocessing, hyperparameter tuning, and evaluation for optimal model performance.

### 3) How to you improve make the model more generalized ?

To improve the generalization of a machine learning model, including the `HistGradientBoostingRegressor`, you can consider several strategies:

1. **Cross-Validation**: Utilize techniques like k-fold cross-validation to assess the model's performance on multiple subsets of the data. This helps in obtaining a more reliable estimate of the model's performance and ensures that it generalizes well to unseen data.

2. **Feature Engineering**: Carefully engineer and select features that are most relevant to the prediction task. This includes identifying and removing irrelevant or redundant features, creating new features that capture useful information, and transforming features to better suit the model assumptions.

3. **Regularization**: Apply regularization techniques such as shrinkage (learning rate) and tree pruning to prevent overfitting. Adjusting the regularization parameters can help control the complexity of the model and improve its ability to generalize.

4. **Hyperparameter Tuning**: Experiment with different hyperparameters of the model, such as the number of trees, maximum depth of trees, and minimum samples per leaf. Grid search or randomized search techniques can be employed to find the optimal combination of hyperparameters that yield the best performance on a validation dataset.

5. **Ensemble Methods**: Consider using ensemble methods like bagging and boosting to combine multiple models trained on different subsets of the data. Ensemble methods often lead to better generalization by reducing the variance of the model predictions.

6. **Data Augmentation (if applicable)**: For certain types of data, such as image or text data, data augmentation techniques can be employed to increase the diversity of the training data. This can help the model generalize better by exposing it to a wider range of variations in the input data.

7. **Early Stopping**: Monitor the model's performance on a validation dataset during training and stop training when the performance starts deteriorating. This prevents overfitting and ensures that the model is not trained for too many iterations, which can lead to memorizing the training data.

8. **Model Selection**: Experiment with different machine learning algorithms and architectures to find the one that best suits the problem at hand. It's essential to choose a model that strikes a balance between complexity and simplicity and can capture the underlying patterns in the data without overfitting.

By implementing these strategies, you can improve the generalization performance of the `HistGradientBoostingRegressor` model and ensure that it performs well on unseen data.


## Acknowledgements


 - [Metro Interstate Traffic](https://www.kaggle.com/code/meemr5/traffic-volume-prediction-time-series-starter/notebook)
 - [Feature importance using Random forest regressor](https://mljar.com/blog/feature-importance-in-random-forest/)
 - [Time Series Analysis](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at karanshah51101@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

