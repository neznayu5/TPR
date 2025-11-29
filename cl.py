import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
df = pd.read_csv(url)

print(df.head())
print(df.info())

if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)
else:
    print("'Cabin' column is already missing.")

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

X = df[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pipeline = Pipeline([
    ('classifier', LogisticRegression(solver='liblinear'))
])

param_grid = [
    {'classifier': [LogisticRegression()], 'classifier__solver': ['liblinear'], 'classifier__penalty': ['l1', 'l2']},
    {'classifier': [DecisionTreeClassifier()], 'classifier__max_depth': [None, 10, 20, 30]},
    {'classifier': [RandomForestClassifier()], 'classifier__n_estimators': [100, 200]}
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print("AUC-ROC:", roc_auc)