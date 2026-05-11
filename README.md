# Olympic Country Efficiency Project

## Overview
This project analyzes Olympic performance at the country level using machine learning and data analysis techniques.  

We transformed an individual athlete Olympic dataset into a country-level dataset and developed several predictive models to estimate Olympic medal counts using both historical performance and economic indicators.

Link to our app!: https://olympic-efficiency-0c2ba2a1a652.herokuapp.com/

---

# Repository Structure

## App/
Contains all files required to run the Flask web application, including:

- HTML templates
- CSS styling
- Python backend files
- Pickle (`.pkl`) files containing saved machine learning models

The app allows users to make Olympic medal predictions using our trained models.

---

## Datasets/
Contains all datasets used throughout the project.

### Included datasets:
- `athlete_events.csv`  
  Individual-level Olympic athlete dataset used as the foundation of the project.

- `gdp.csv`  
  Contains GDP data for countries across different years.

- `hdi.csv`  
  Contains Human Development Index (HDI) values by country and year.

- `population.csv`  
  Contains population data for countries across different years.

- `noc_regions.csv`  
  Mapping file containing Olympic country abbreviations (NOC codes).

- `olympic_country_year_features.csv`  
  The country-level dataset created for this project by combining Olympic, GDP, population, and HDI data.

---

## Formatted_Notebook.ipynb
Main Jupyter notebook containing:

- Data loading and cleaning
- Data visualization
- Feature engineering
- Model creation and evaluation
- Cross-validation results
- Machine learning experiments

---

## Olympic_Efficiency.pptx
PowerPoint presentation used to present the project findings and methodology.

---

## abstract.txt
Project abstract summarizing the goals, methods, and findings of the project.

---

## make_new_csv.py
Python script used to generate the country-level Olympic dataset.

This script:
- Processes the individual athlete dataset
- Aggregates Olympic results to the country-year level
- Merges economic and demographic datasets
- Creates the final feature dataset used for modeling

---

# Machine Learning Models

We experimented with several machine learning models, including:

- Linear Regression
- Random Forest
- XGBoost
- Neural Networks

---

# Main Features Used

## Performance-Based Model
- Previous medals won
- Athletes sent

## Efficiency Model
- GDP per capita
- Population
- Human Development Index (HDI)

---




