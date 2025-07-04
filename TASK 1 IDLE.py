#Importing Required modules
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time

#Loading Dataset with Dask.
file_path = "D:/CODETECH INTERN/Lung Cancer.csv"
df = dd.read_csv(file_path, assume_missing=True)
print("Data View:/n",df.head())

#Convert Date columns
df['diagnosis_date'] = dd.to_datetime(df['diagnosis_date'], errors='coerce')
df['end_treatment_date'] = dd.to_datetime(df['end_treatment_date'], errors='coerce')

#Start Timer
start = time.time()

#Big Data Analysis
# 1. Survival rate by cancer stage
survival_by_stage = df.groupby('cancer_stage')['survived'].mean().compute()

# 2. Average BMI and cholesterol by country
bmi_chol_by_country = df.groupby('country')[['bmi', 'cholesterol_level']].mean().compute().sort_values(by='bmi', ascending=False)

# 3. Comorbidity prevalence by smoking status
comorbidities = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer']
comorbidity_by_smoking = df.groupby('smoking_status')[comorbidities].mean().compute()

# 4. Treatment duration (in days) vs survival correlation
df['treatment_days'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
survival_corr_data = df[['treatment_days', 'survived']].dropna().compute()
correlation = survival_corr_data.corr().loc['treatment_days', 'survived']

#End Timer
end = time.time()

#Insights or Result
print("Survival Rate by Cancer Stage:\n",survival_by_stage)
print("\nAvg BMI and Cholesterol by Country:\n", bmi_chol_by_country.head(10))
print("\nComorbidities by Smoking Status:\n", comorbidity_by_smoking)
print(f"\nCorrelation between Treatment Duration and Survival: {correlation:.3f}")
print(f"\nAnalysis completed in {end - start:.2f} seconds")

#Plot showing Survival rate based on the stage
plt.figure(figsize=(6, 5))
plt.bar(survival_by_stage.sort_index().index, survival_by_stage.sort_index().values,color='skyblue')
plt.title("Survival Rate by Cancer Stage")
plt.ylabel("Survival Rate")
plt.xlabel("Cancer Stage")
plt.tight_layout()
plt.show()

