# Lifestyle Factors and General Health: A Predictive Analysis

## Description

This project investigates the relationship between various lifestyle factors and self-assessed general health status using the comprehensive 2022 Behavioral Risk Factor Surveillance System (BRFSS) dataset. Key aspects of the project include comprehensive dataset analysis, rigorous data preprocessing, the application of predictive models, elaborate visualizations, and strategic health insights.

## Data Source

The analysis is based on the [2022 Behavioral Risk Factor Surveillance System (BRFSS) dataset](https://www.cdc.gov/brfss/annual_data/annual_2022.html), provided by the Centers for Disease Control and Prevention (CDC). This dataset includes responses from a wide cross-section of the population across the United States and its territories.

## Table of Contents

- [Description](#description)
- [Data Source](#data-source)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Running the Code](#running-the-code)
- [Methodology](#methodology)
- [Findings](#findings)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the project environment and run the analysis:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/general-health-lifestyle-factors
cd general-health-lifestyle-factors
pip install -r requirements.txt
```
## Repository Structure

- `README.md`: Provides an overview and instructions for the project.
- `Data Cleaning.py`: Contains the data cleaning and preprocessing steps.
- `Baseline Models.py`: Script for developing baseline predictive models.
- `RF and Boosting.py`: Implements the initial Random Forest and Gradient Boosting models.
- `RF and Boosting2.py`: Second iteration of tuning for Random Forest and Gradient Boosting models.
- `RF and Boosting3.py`: Final iteration for advanced tuning and evaluation of the models.
- `Paper.pdf`: The comprehensive research paper detailing background, methodology, analysis, and conclusions.

## Running the Code

1. Clone this repository to your local machine.
2. Install the required Python packages using: `pip install -r requirements.txt`.
3. Execute the Python scripts in the following sequence to replicate the analysis:
   - `python "Data Cleaning.py"`: Cleans and prepares the dataset for analysis.
   - `python "Baseline Models.py"`: Develops and evaluates baseline predictive models.
   - `python "RF and Boosting.py"`: Applies and evaluates the initial Random Forest and Gradient Boosting models.
   - For further tuning and evaluation of the models, run `python "RF and Boosting2.py"` and `python "RF and Boosting3.py"` as needed.

## Methodology

The project employs comprehensive statistical analysis and data mining techniques such as Regression Analysis, Decision Trees, Random Forest, and Gradient Boosting to predict general health outcomes based on lifestyle factors. Detailed methodologies including data cleaning, preprocessing, and model development are extensively documented within the paper and corresponding Python scripts.

## Findings

The analysis demonstrates significant correlations between lifestyle factors and self-assessed general health, emphasizing the potential of Random Forest and Gradient Boosting models in public health strategies. The tuned Gradient Boosting model, with an accuracy of 85.79%, effectively predicts general health status, underscoring the impact of lifestyle factors.

## Conclusion

The findings highlight the critical role of lifestyle choices on general health and showcase the capability of machine learning models to forecast health outcomes. Insights from this study provide a solid foundation for designing health interventions aimed at enhancing general well-being.

## Contributing

Contributions, suggestions, and feedback are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is released under the MIT License. Please refer to the LICENSE file for more details.

## Acknowledgments

- Georgia Institute of Technology for providing the educational platform.
- Centers for Disease Control and Prevention (CDC) for making the BRFSS dataset publicly available.

