# Titanic-Kaggle
Model for Kaggle Titanic Challenge: https://www.kaggle.com/competitions/titanic/overview


## Data
### Files
Data files are located in the dataset directory
- gender_submission.csv: a sample submission file provided by kaggle
- test.csv: test set for final prediction. No target (Survived) values provided
- train.csv: training set including target (Survived) values

### Provided Columns
|Column|Description|
|---|---|
|PassengerId|Index|
|Survived|Passenger Survived|
|Pclass|Ticket Class|
|Name|Name of Passenger|
|Sex|Sex of Passenger|
|Age|Age of Passenger|
|SibSp|# of siblings/spouses aboard|
|Parch|# of parents/children aboard|
|Ticket|Ticket Number|
|Fare|Passenger Fare|
|Cabin|Cabin Number|
|Embarked|Port of Embarkation|

### Engineered Features
|Column|Description|
|---|---|
|ticketNo|Numerical portion of ticket number|
|ticketTxt|Textual portion of ticket number|
|CabinLvl| Alphabetical cabin level|
|cabinNo|Numerical portion of cabin number|
|surname|Surname of passenger|
|title|Title of passenger|

## Model
Gradien Boosting Classifier