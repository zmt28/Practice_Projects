import streamlit as st
from sklearn import datasets

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

import numpy as np
import matplotlib.pyplot as plt

st.title("Streamlit Machine Learning WebApp")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', "Wine dataset"))

classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))


# load the chosen Data
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        P = st.sidebar.slider("P", 1.0, 5.0)
        params["K"] = K
        params["P"] = P
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
        G = st.sidebar.slider("Gamma", 1, 20)
        params["G"] = G
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        min_samp_split = st.sidebar.slider("min_samples_split", .1, 1.0)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["min_samp_split"] = min_samp_split
    return params


params = add_parameter_ui(classifier_name)
normalizer = Normalizer()
if classifier_name == 'Random Forest':
    normalized = False
else:
    normalized = True


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"], p=params["P"])
        normalizer.fit_transform(X=X)
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], gamma=params["G"])
        normalizer.fit_transform(X=X)
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], min_samples_split=params["min_samp_split"],
                                     random_state=17)
    return clf


clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")
st.write(f"normalized? = {normalized}")

# PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("principal Component 1")
plt.ylabel("principal Component 2")
plt.colorbar()

# plt show
st.pyplot()

# TODO
# add more parameters to models
# add more/other classifiers
