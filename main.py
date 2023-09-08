# %% [markdown]
# ### Importing Libraries

# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### Reading the Data

# %%
data = pd.read_excel("customer_churn_large_dataset.xlsx")

# %%
data.head()

# %%
data.shape

# %% [markdown]
# ### Checking for Null Values and datatypes

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# ### Removing unwanted Columns

# %%
data.drop(["CustomerID","Name"],axis=1,inplace=True)

# %% [markdown]
# ### Dividing the data in Positive And Negative Churn

# %%
df1=data[data["Churn"]==0]
df2=data[data["Churn"]==1]

# %%
df1

# %%
df2

# %% [markdown]
# ### Data Visualization

# %%
tenure_churn_no = df1.Subscription_Length_Months
tenure_churn_yes = df2.Subscription_Length_Months

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([tenure_churn_yes, tenure_churn_no], color=['skyblue','yellow'],label=['Churn=Yes','Churn=No'])
plt.legend()

# %%
mc_churn_no = df1.Monthly_Bill   
mc_churn_yes = df2.Monthly_Bill    

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([mc_churn_yes, mc_churn_no], color=['skyblue','yellow'],label=['Churn=Yes','Churn=No'])
plt.legend()

# %%
data.head()

# %%
f=data.groupby("Churn")["Location"].value_counts()
f


# %% [markdown]
# ### Train Test Split

# %%
X=data.drop("Churn",axis=1)
y=data["Churn"]

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# %% [markdown]
# ### Importing more Liraries

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_validate

# %% [markdown]
# ### Encoding And Scaling the data

# %%
categorical_features=['Gender','Location']
numerical_features=['Age','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']

one_hot_encoder=OneHotEncoder()
standard_scaler=StandardScaler()

preprocessor=ColumnTransformer(transformers=(
    ('encode_gender',one_hot_encoder,categorical_features),
    ('standardization',standard_scaler,numerical_features)
))

# %%
preprocessor

# %% [markdown]
# ### Checking for the Best Model

# %% [markdown]
# ### Logistic Regression

# %%
clf=Pipeline(steps=(
    ('preprocessing',preprocessor),
    ('classifier',LogisticRegression())
))

clf.fit(X_train,y_train)
print("Accuracy score of Logistic Regression is: ",clf.score(X_test,y_test))

y_pred=clf.predict(X_test)

print("The precision score of Logistic Regression is: ",precision_score(y_test,y_pred))
print("The recall score of Logistic Regression is: ",recall_score(y_test,y_pred))
print("The F1 score of Logistic Regression is: ",f1_score(y_test,y_pred))


# %% [markdown]
# ### KNeighbors Classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

clf1=Pipeline(steps=(
    ('preprocessing',preprocessor),
    ('classifier',KNeighborsClassifier())
))

clf1.fit(X_train,y_train)
print("Accuracy score of Logistic Regression is: ",clf1.score(X_test,y_test))

y_pred=clf1.predict(X_test)

print("The precision score of KNeighbors Classifier is: ",precision_score(y_test,y_pred))
print("The recall score of KNeighbors Classifier is: ",recall_score(y_test,y_pred))
print("The F1 score of KNeighbors Classifier is: ",f1_score(y_test,y_pred))

# %% [markdown]
# ### Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB

clf2=Pipeline(steps=(
    ('preprocessing',preprocessor),
    ('classifier',GaussianNB())
))

clf2.fit(X_train,y_train)
print("Accuracy score of GaussianNB is: ",clf2.score(X_test,y_test))

y_pred=clf2.predict(X_test)

print("The precision score of Naive Bayes is: ",precision_score(y_test,y_pred))
print("The recall score of Naive Bayes is: ",recall_score(y_test,y_pred))
print("The F1 score of Naive Bayes is: ",f1_score(y_test,y_pred))

# %% [markdown]
# ### Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

clf3=Pipeline(steps=(
    ('preprocessing',preprocessor),
    ('classifier',DecisionTreeClassifier())
))

clf3.fit(X_train,y_train)
print("Accuracy score of Decision Tree Classifier is: ",clf.score(X_test,y_test))

y_pred=clf3.predict(X_test)

print("The precision score of Decision Tree Classifier is: ",precision_score(y_test,y_pred))
print("The recall score of Decision Tree Classifier is: ",recall_score(y_test,y_pred))
print("The F1 score of Decision Tree Classifier is: ",f1_score(y_test,y_pred))

# %% [markdown]
# ### Random Forest Classifier

# %%
clf4=Pipeline(steps=(
    ('preprocessing',preprocessor),
    ('classifier',RandomForestClassifier())
))

clf4.fit(X_train,y_train)
print("Accuracy score of Random Forest Classifier is: ",clf4.score(X_test,y_test))

y_pred=clf4.predict(X_test)

print("The precision score of Random Forest Classifier is: ",precision_score(y_test,y_pred))
print("The recall score of Random Forest Classifier is: ",recall_score(y_test,y_pred))
print("The F1 score of Random Forest Classifier is: ",f1_score(y_test,y_pred))

# %% [markdown]
# ### K-Fold Cross Validation 
# using this technique to choose best model

# %%
from sklearn.model_selection import cross_val_score

# %%
print("Logistic Regression: ",cross_val_score(clf, X, y,cv=3))
print("KNeighbors: ",cross_val_score(clf1, X, y,cv=3))
print("Naive Bayes: ",cross_val_score(clf2, X, y,cv=3))
print("Decision Tree: ",cross_val_score(clf3, X, y,cv=3))
print("Random Forest: ",cross_val_score(clf4, X, y,cv=3))

# %% [markdown]
# ### My model (Logistic Regression)

# %% [markdown]
# #### Training the model

# %%
clf.fit(X_train,y_train)

# %%
clf.score(X_train,y_train)

# %%
y_pred=clf.predict(X_test)

# %% [markdown]
# #### Getting the precision,recall and f1 score using classification report

# %%
from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))

# %% [markdown]
# ### Confusion Matrix

# %%
import seaborn as sn
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# %% [markdown]
# #### Accuracy

# %%
round((2816+2196)/(2816+2196+2729+2259),2)


