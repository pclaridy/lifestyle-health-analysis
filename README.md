# **Lifestyle Factors and General Health: Predicting Wellness Through Behavioral Data**

## **Table of Contents**
1. [Problem Statement](#1-problem-statement)  
2. [Data Source](#2-data-source)  
3. [Data Cleaning & Preprocessing](#3-data-cleaning--preprocessing)  
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)  
5. [Modeling Approach](#5-modeling-approach)  
6. [Evaluation Metrics](#6-evaluation-metrics)  
7. [Outcome](#7-outcome)  
8. [Tools Used](#8-tools-used)  
9. [Business Impact / Use Case](#9-business-impact--use-case)

---

## **1. Problem Statement**  
Understanding what drives individuals to report poor general health is critical for designing effective public health interventions and outreach strategies. This project builds predictive models to classify self-reported general health status based on behavioral, demographic, and lifestyle factors using survey data from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS). Features such as physical activity, chronic conditions, alcohol consumption, and smoking habits are analyzed to uncover which patterns most strongly correlate with health perceptions.

This study focuses on general health as the response variable rather than a specific disease like heart disease. The choice allows for a broader understanding of health and captures the cumulative impact of lifestyle. General health ratings are valuable indicators of underlying chronic issues, perceived wellness, and future health risks. If someone exhibits a certain combination of lifestyle behaviors and demographics, the model can predict how they’re likely to rate their general health. That kind of insight helps identify at-risk groups and supports more proactive, population-level health planning.

---

## **2. Data Source**  
The dataset comes from the [2022 Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_2022.html), maintained by the CDC. It includes over 400,000 survey responses collected across the United States and its territories. For this project, the selected variables include:

- General health status (target)  
- Physical activity  
- Smoking status  
- Alcohol intake  
- Presence of asthma or heart disease  
- Body Mass Index (BMI)  
- Demographics like age, gender, race, income, and education level

Cleaned dataset available here: [cleaned_data.csv](https://github.com/pclaridy/lifestyle-health-analysis/blob/main/cleaned_data.csv)

---

## **3. Data Cleaning & Preprocessing**  
Steps in the data pipeline included:

- Extracting features of interest from the raw `.XPT` file  
- Renaming columns for clarity  
- Dropping missing values and filtering invalid entries  
- Removing outliers using z-scores  
- Encoding categorical variables for modeling  
- Splitting the cleaned dataset into training and test sets

These steps ensured consistency and model-readiness across all phases of analysis.

---

## **4. Exploratory Data Analysis (EDA)**  
Exploratory analysis included:

- Examining class balance in the target variable  
- Summary statistics across features  
- Correlation checks  
- Initial insights into feature relationships with general health

This phase helped identify which variables were most likely to influence the model and where to focus feature selection efforts. No visuals were included in this version of the project.

---

## **5. Modeling Approach**

The project followed a two-phase modeling strategy:

### **Baseline Models**
- K-Nearest Neighbors (KNN)  
- Logistic Regression  
- Decision Tree  

These models offered quick insights and established a reference point for performance.

### **Advanced Models**
- Random Forest  
- Gradient Boosting  

Both were tuned using GridSearchCV and RandomizedSearchCV to improve predictive accuracy. Key parameters tuned included:

- n_estimators  
- max_depth  
- min_samples_split  
- min_samples_leaf  
- learning_rate  
- max_features  

Scripts were modular and documented for reproducibility. Available here:
- [RF and Boosting.py](https://github.com/pclaridy/lifestyle-health-analysis/blob/main/RF%20and%20Boosting.py)  
- [RF and Boosting2.py](https://github.com/pclaridy/lifestyle-health-analysis/blob/main/RF%20and%20Boosting2.py)  
- [RF and Boosting3.py](https://github.com/pclaridy/lifestyle-health-analysis/blob/main/RF%20and%20Boosting3.py)

---

## **6. Evaluation Metrics**  
To compare model performance, the following metrics were used:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1 Score (weighted)  

These provided a balanced view of how well each model handled class predictions, especially with imbalanced labels.

---

## **7. Outcome**

| Model                  | Accuracy (Pre-Tuning) | Accuracy (Post-Tuning) |
|------------------------|-----------------------|-------------------------|
| K-Nearest Neighbors    | 84.22%                | —                       |
| Logistic Regression    | 85.34%                | —                       |
| Decision Tree          | 81.15%                | —                       |
| Random Forest          | 83.53%                | 85.64%                  |
| Gradient Boosting      | 85.61%                | **85.79%**              |

The tuned Gradient Boosting model performed the best, achieving 85.79% accuracy on the test set. This accuracy reflects the model’s ability to correctly predict how individuals rate their own general health using inputs such as physical activity, chronic conditions, BMI, and demographic data.

In other words, the model can assess someone’s health perception based on lifestyle and background factors. That capability helps reveal how these variables work together to shape well-being and offers a practical approach to identifying groups that may need health interventions.

---

## **8. Tools Used**

- **Languages**: Python  
- **Libraries**: pandas, scikit-learn, numpy  
- **Modeling**: KNN, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting  
- **Tuning**: GridSearchCV, RandomizedSearchCV  
- **Preprocessing**: Label encoding, z-score filtering, stratified train/test split  
- **Environment**: Local Python setup using modular scripts  

**Documentation**:  
A full research paper is included: [Paper.pdf](https://github.com/pclaridy/lifestyle-health-analysis/blob/main/Paper.pdf)

---

## **9. Business Impact / Use Case**

This project demonstrates how lifestyle and demographic data can be used to predict health perceptions at the individual level. These predictions can guide:

- Healthcare providers in early identification of at-risk patients  
- Public health agencies in shaping community wellness campaigns  
- Policy makers looking to invest in preventative care initiatives  
- Insurance companies exploring personalized wellness strategies

The ability to model general health status from modifiable factors offers a powerful tool for improving population health. Insights like these can be used to develop interventions that are proactive, targeted, and informed by real-world data. They also highlight the broader implication that lifestyle choices play a major role in how people perceive their health, making general well-being a meaningful focus for public health strategy.
