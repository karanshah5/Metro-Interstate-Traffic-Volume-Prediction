# Metro-Interstate-Traffic-Volume-Prediction

Utilized Pandas, Matplotlib, Seaborn, NumPy, and Scikit-learn for comprehensive data analysis, visualization, preprocessing, and predictive modeling to forecast traffic volume.

Data directory contains a train set and a test set we were given by 123 of AI during our hackathon. If you want the raw data that I created these files from, check out here: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

## Project Methodology

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

- Calculating and visualizing the correlation matrix between numerical features and the target variable ('traffic_volume').

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

## Acknowledgements

 - [Metro Interstate Traffic Prediction](https://www.kaggle.com/code/meemr5/traffic-volume-prediction-time-series-starter/notebook)
 - [Feature importance using Random forest regressor](https://mljar.com/blog/feature-importance-in-random-forest/)
 - [Time Series Analysis](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at karanshah51101@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

