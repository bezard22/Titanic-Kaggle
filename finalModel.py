import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier

train_path = r"dataset\train.csv"
test_path = r"dataset\test.csv"
train_df= pd.read_csv(train_path)
test_df= pd.read_csv(test_path)


# Feature Engineering
def ticketNo(ticketStr):
    if isinstance(ticketStr, str):
        noStr = re.findall(r"[0-9]*$", ticketStr)[0]
        if noStr.isnumeric():
            return int(noStr)
    return np.nan

def ticketTxt(ticketStr):
    if isinstance(ticketStr, str):
        noStr = re.findall(r"[0-9]*$", ticketStr)[0]
        return ticketStr[:-len(noStr)]
    return np.nan

def cabinNo(cabinStr):
    if isinstance(cabinStr, str):
        return re.findall(r"[0-9]*$", cabinStr)[0]
    return np.nan

def cabinLvl(cabinStr):
    if isinstance(cabinStr, str):
        return re.findall(r"^[A-Z]", cabinStr)[0]
    return np.nan

def title(name):
    if isinstance(name, str):
        return re.findall(r"\b\w*\.", name)[0]
    return np.nan

train_df["ticketNo"] = train_df["Ticket"].apply(ticketNo)
train_df["ticketTxt"] = train_df["Ticket"].apply(ticketTxt)
train_df["cabinLvl"] = train_df["Cabin"].apply(cabinLvl)
train_df["cabinNo"] = train_df["Cabin"].apply(cabinNo)
train_df["title"] = train_df["Name"].apply(title)

test_df["ticketNo"] = test_df["Ticket"].apply(ticketNo)
test_df["ticketTxt"] = test_df["Ticket"].apply(ticketTxt)
test_df["cabinLvl"] = test_df["Cabin"].apply(cabinLvl)
test_df["cabinNo"] = test_df["Cabin"].apply(cabinNo)
test_df["title"] = test_df["Name"].apply(title)

y_pred = pd.DataFrame()
y_pred["PassengerId"] = test_df["PassengerId"]
# Pre-Processing
train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

for col in train_df.columns:
    if not pd.api.types.is_numeric_dtype(train_df[col]):
        train_df[col] = train_df[col].map({key: val for val, key in enumerate(train_df[col].unique())})
        test_df[col] = test_df[col].map({key: val for val, key in enumerate(test_df[col].unique())})

X_train = train_df.drop(columns=["Survived"])
X_test = test_df
y_train = train_df["Survived"]

preProcessor = make_pipeline(StandardScaler(), SimpleImputer())
X_train = preProcessor.fit_transform(X_train)
X_test = preProcessor.transform(X_test)

# Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred["Survived"] = model.predict(X_test)
y_pred.to_csv(r"finalPrediction.csv", index=False)

