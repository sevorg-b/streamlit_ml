import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

st.title("Machine learning and medical diagnosis")

df = pd.read_csv('data/BRCA.csv')
data = df.copy()
st.write(
    "This dataset consists of a group of breast cancer patients, who had surgery to remove their tumour. The dataset consists of the following variables:")
st.write(data.head())

# Data and feature engineering
df = df.drop([334, 335, 336, 337, 338, 339, 340])
data = data.drop([334, 335, 336, 337, 338, 339, 340])
df = df.drop('Date_of_Last_Visit', axis=1)
df['Patient_Status'] = df['Patient_Status'].replace('Dead', 'Deceased')

ohe = pd.get_dummies(data[['Gender', 'Tumour_Stage', 'Histology', 'ER status', 'PR status',
                           'HER2 status', 'Surgery_type']])

data = data.drop(['Gender', 'Patient_ID', 'Tumour_Stage', 'Histology', 'ER status', 'PR status',
                  'HER2 status', 'Surgery_type', 'Date_of_Surgery', 'Date_of_Last_Visit'], axis = 1)

le = LabelEncoder()
data['patient_status'] = le.fit_transform(data['Patient_Status'])
data = data.drop('Patient_Status', axis = 1)

data = data.join(ohe)

data['er_status_positive'], data['pr_status_positive'], data['her2_status_negative'], data['her2_status_positive'] = \
    data['ER status_Positive'], data['PR status_Positive'], data['HER2 status_Negative'], data['HER2 status_Positive']

data = data.drop(['ER status_Positive',
                  'PR status_Positive', 'HER2 status_Negative', 'HER2 status_Positive'], axis = 1)
data = data.drop(['er_status_positive', 'pr_status_positive'], axis = 1)

cut = data[data['patient_status'] <= 1]
data = cut

# Missing data
st.title("Reviewing our dataset")
st.write("Let's explore our data to see if anything is missing. Before any analysis can begin we need to ensure data is of sufficient quality. As the strength of our prediction will be reflected in the quality of our data!")

def missingdata():
    plt.figure(figsize=(10, 10), dpi = 250)
    g = sns.heatmap(data.isnull(), cmap='RdBu')
    g.set_xlabel("Features")
    g.set_ylabel("Index")
    g.set_title('Missing feature data')
    st.pyplot(g.figure)

missingdata()

st.write("Great, no data is missing!")


# Univariate analysis

st.text("")
st.write(
    "From these data we want to predict the patient status (aka the target variable). Let's look at the target variable in detail:")


def countplot ():
    plt.figure(figsize = (10, 5), dpi = 250)
    p = sns.countplot(df['Patient_Status'])
    p.set_xlabel("Patient status")
    p.set_ylabel("Count")
    st.pyplot(p.figure)


countplot()

st.text("")
st.title("Basic descriptive analysis")
st.write("Using basic descriptive statistics we can generate basic insights into our data!")
st.write(data.describe())

# Correlation analysis
st.text("")
st.write(
    "To explore how our data correlations we can call the Pandas internal correlation function. This function takes three method arguments, so feel free to explore how correlations change per method!")
methods = ['Spearman', 'Pearson', 'Kendall']
selection = st.selectbox('Please select correlation method:', methods)

if selection == 'Pearson':
    i = 'pearson'
elif selection == 'Spearman':
    i = 'spearman'
else:
    i = 'kendall'

st.write(data.corr(method = i))

# Model development
st.write(
    "After some basic data and feature engineering (that I'll spare you from!) we can start building a basic ML model to set our baseline performance. First, we must define our training set:")

X = data.drop('patient_status', axis = 1)
y = data['patient_status']

st.write(X)
st.text("")
st.write("And the target variable we are trying to predict:")
st.write(y)

st.write("Now we have our data defined, we'll use a selection of models and see which performs best out the box! For this we'll need classificatgion algorithms, let's see how they perform!")

algorithms = ['Logistic regression', 'Kneighbours classifier', 'Random Forest Classifier']
selection = st.selectbox('Please select correlation method:', algorithms)

if selection == 'Logistic Regression':
    i = LogisticRegression
elif selection == 'Kneighbours classifier':
    i = KNeighborsClassifier
else:
    i = RandomForestClassifier
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


def modeldev(i, j, k, l, m):
    model = i()
    model.fit(j, k)
    y_pred = model.predict(l)
    rep = classification_report(m, y_pred, output_dict = True)
    rep = pd.DataFrame(rep)
    rep = rep.drop('support', axis=0)
    st.write(rep)

modeldev(i, X_train, y_train, X_test, y_test)

st.write("Here we are, results from our trained model!")




# st.write(
#     "One parameter we can look to change is K, the value representing the count of nearest neighbours, and its value is vital to developing a model with good classification capability. ")


# error_rate = []
# for i in range(1,40):
#  knn = KNeighborsClassifier(n_neighbors=i)
#  knn.fit(X_train,y_train)
#  pred_i = knn.predict(X_test)
#  error_rate.append(np.mean(pred_i != y_test))
#
# optimalk = pd.DataFrame({
#     'k': range(1,40),
#     'error_rate': error_rate
# })
# plt.figure(figsize = (10, 10), dpi = 200)
# plt.title("Error rate by value of K")
# plt.ylabel("Error rate")
# plt.xlabel("Value of K")
# p = sns.lineplot(range(1, 40), optimalk['error_rate'], markers = True)
# st.text("")
# st.pyplot(p.figure)
# st.write("As we can see the minimum error is: {} at K = {}".format(min(error_rate), error_rate.index(min(error_rate))))