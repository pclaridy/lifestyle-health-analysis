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
Understanding what drives individuals to report poor general health is critical for designing effective public health interventions and outreach strategies. This project aims to develop predictive models that classify self-reported general health status based on behavioral, demographic, and lifestyle factors using survey data from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS). Features such as physical activity, chronic conditions, alcohol consumption, and smoking habits are analyzed to uncover which patterns most strongly correlate with poor health perceptions. By focusing on both model accuracy and interpretability, the analysis supports data-driven public health planning that can guide early intervention efforts and resource allocation.

---

## **2. Data Source**  
The dataset is sourced from the [2022 Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_2022.html), maintained by the CDC. The dataset includes over 400,000 survey responses covering adults across the United States and its territories. Variables used in this project include:

- General health status (target)  
- Smoking status  
- Alcohol intake  
- Physical activity  
- Chronic conditions (asthma, heart disease)  
- Body Mass Index (BMI)  
- Demographics (age, race, gender, education, income)

---

## **3. Data Cleaning & Preprocessing**  
The data was processed using the following pipeline:

- Extracted features of interest from `.XPT` format  
- Renamed columns for interpretability  
- Removed missing values and invalid ranges  
- Applied z-score based outlier removal  
- Mapped categorical values into interpretable labels  
- Filtered extreme alcohol consumption outliers  
- Encoded variables numerically for modeling  
- Split data into training and test sets

The cleaned dataset was saved and used consistently throughout model development.

---

## **4. Exploratory Data Analysis (EDA)**  
Exploratory insights included:

- Distribution checks across the target variable  
- Summary statistics of predictors  
- Correlation analysis of numerical features  
- Balance checks between class groups  

These steps guided the selection of predictive features and preprocessing strategies. No visual elements were produced for this project.

---

## **5. Modeling Approach**

The project employed a two-phase modeling strategy:

### **Baseline Models**
- **K-Nearest Neighbors (KNN)**  
- **Logistic Regression**  
- **Decision Tree**

These models provided quick, interpretable results and established a baseline for performance comparison.

### **Advanced Models**
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**

Each model was initially trained with default parameters, followed by extensive hyperparameter tuning.

### **Tuning Approach**
Used a combination of GridSearchCV and RandomizedSearchCV to tune:

- `n_estimators`  
- `max_depth`  
- `min_samples_split`  
- `min_samples_leaf`  
- `learning_rate`  
- `max_features`

Scripts were modular, and tuning iterations were saved in sequenced files (`RF and Boosting.py`, `RF and Boosting2.py`, `RF and Boosting3.py`).

---

## **6. Evaluation Metrics**

All models were evaluated using:

- **Accuracy**  
- **Precision (Weighted)**  
- **Recall (Weighted)**  
- **F1 Score (Weighted)**

The top-performing models — Gradient Boosting and Random Forest — were compared across these metrics before and after tuning.

---

## **7. Outcome**

| Model                  | Accuracy (Pre-Tuning) | Accuracy (Post-Tuning) |
|------------------------|-----------------------|-------------------------|
| K-Nearest Neighbors    | 84.22%                | —                       |
| Logistic Regression    | 85.34%                | —                       |
| Decision Tree          | 81.15%                | —                       |
| Random Forest          | 83.53%                | 85.64%                  |
| Gradient Boosting      | 85.61%                | **85.79%**              |

The best performing model was the tuned **Gradient Boosting Classifier**, which achieved 85.79% accuracy and consistently strong weighted precision and recall. This performance confirms the benefit of ensemble learning in modeling lifestyle-related health outcomes.

---

## **8. Tools Used**

- **Languages**: Python  
- **Libraries**: pandas, scikit-learn, numpy  
- **Modeling**: KNN, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting  
- **Tuning**: GridSearchCV, RandomizedSearchCV  
- **Preprocessing**: Label encoding, z-score filtering, stratified data splitting  
- **Environment**: Local Python environment with modular scripts  

**Documentation & Reporting**:
- A comprehensive research paper (`Paper.pdf`) accompanies this project. It outlines the background, methodology, modeling strategies, and evaluation results.

---

## **9. Business Impact / Use Case**

The ability to predict health outcomes using lifestyle data enables:

- **Healthcare organizations** to identify at-risk populations  
- **Public health campaigns** to target interventions based on behavioral profiles  
- **Insurance providers** to offer proactive wellness strategies  
- **Policy developers** to explore how lifestyle data can inform community health investments  

This model demonstrates how machine learning can contribute to cost-effective and targeted public health responses.
