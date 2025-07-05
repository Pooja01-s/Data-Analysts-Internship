# Data-Analysts-Internship
This project involves four interconnected tasks focused on lung cancer data analysis, predictive modeling, and sentiment analysis, enhanced by visual dashboards. The tasks span across big data processing, machine learning, natural language processing (NLP), and business intelligence.
1. Big Data Analytics using Dask
Objective: Perform scalable exploratory data analysis on a large lung cancer dataset.
What Was Done:
Loaded the dataset using Dask to handle large-scale data efficiently.
Converted date columns (diagnosis_date, end_treatment_date) for time-based calculations.
Analyzed:
Survival rate by cancer stage
Average BMI and cholesterol levels by country
Comorbidity prevalence by smoking status
Correlation between treatment duration and survival
Visualized survival rate by stage using a bar chart.
Measured and printed total execution time.
2. Machine Learning Model for Survival Prediction
Objective: Build a predictive model to classify whether a lung cancer patient survived.
What Was Done:
Loaded the dataset using Pandas.
Dropped irrelevant or redundant columns (id, diagnosis_date, end_treatment_date).
Removed rows with missing data.
Encoded categorical variables using Label Encoding.
Split the data into training and testing sets.
Trained a RandomForestClassifier to predict survival outcomes.
Evaluated performance using:
Accuracy score
Classification report
Confusion matrix heatmap for visual analysis
3. Power BI Report
Objective: Visualize key insights from the lung cancer dataset in an interactive dashboard.
What Was Done (based on context):
Created interactive dashboards using Power BI.
Likely visualizations include:
Survival trends by cancer stage or country
Comorbidities and lifestyle factors
Geographical distribution of health metrics
Provides a business intelligence layer for decision-makers and non-technical users.
3. Sentiment Analysis on Twitter Data
Objective: Analyze sentiment (negative, neutral, positive) in tweets using natural language processing.
What Was Done:
Loaded a dataset of tweets.
Cleaned the text data: removed URLs, punctuation, and lowercased the content.
Used TF-IDF vectorization to transform text into numerical features.
Trained a Logistic Regression model for multi-class classification.
Evaluated results with:
Classification report
Confusion matrix heatmap
Identified and displayed the top 10 influential words for each sentiment class.



