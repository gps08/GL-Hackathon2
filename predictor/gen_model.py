from collections import Counter
from numpy import mean, std
import numpy as np
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score, f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
from joblib import dump

def load_train_set(data_file):
    #Load the data using pandas read_csv method
    df=read_csv(data_file,nrows=40000, sep=",")
    #Display the first few rows in the dataframe
    print(df.head())
    return df

def preprocess_data(df):
    ## Removing all null values of violating locations as it is necessary
    df = df[df['Violation Location'].notnull()]  
    ##Preprocessing the data removing missing values and extremely high parametes
    df = df.drop(["House Number","Summons Number","Street Code1","Street Code2","Street Code3","Issuer Code","Issuer Command","Issuer Squad","Violation Post Code","Vehicle Expiration Date"], axis = 1)
    df = df.dropna(how='any',axis='columns')
    print(f"U sing {len(df)} datapoints for prediction after removing outliers")
    return df

def encoding_categorical_values(x,y):
    # select categorical features for one hot encoding
    cat_ix = x.select_dtypes(include=['object','int']).columns
    print(cat_ix)
    # one hot encode categorical features only
    ct = ColumnTransformer([('o',OneHotEncoder(),cat_ix)], remainder='passthrough')
    x = ct.fit_transform(x)
    return x,y

def data_split(df):
    ##Splitting inputs and outputs
    df = pd.DataFrame(df)
    X, Y = df.drop(["Violation Location"], axis=1), df["Violation Location"]
    print(Y)
    X, Y = encoding_categorical_values(X, Y)
    print('=====\n', X)
    ##Splitting Data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    return [X_train,X_test,y_train,y_test]

def clf_alg(classifier,X_train,y_train):
    if classifier=='DTC':
        clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2)
    elif classifier=='SVM':
        clf = SVC(kernel='poly', degree=3, max_iter=300000)
    elif classifier=='KNN':
        clf = KNeighborsClassifier(metric='manhattan')
    return clf.fit(X_train,y_train)

def model_performance(model_alg,X_test,y_test):
    
    #Predicting for test data
    y_pred = model_alg.predict(X_test)
    precision = precision_score(y_test,y_pred, average='micro')
    print(f"Precision: {precision}")
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Accuracy: {accuracy}")
    recall = recall_score(y_test,y_pred, average='micro')
    print(f"Recall: {recall}")
    f1 = f1_score(y_test,y_pred, average='macro')
    print(f"F1-score: {f1}")
    ##Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    ## Heat Map
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    #sns.set(font_scale=1) # for label size
    #Display the confusion matrix in the form of heatmap
    #sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    #Show the plot

def model_training(df):
    X_train,X_test,y_train,y_test=data_split(df)
    choose_alg = input("Choose Classifier\n DTC-DecisionTree\n SVM-SupportVectorMachines\n KNN-KNearestNeighbour\n")
    model_alg = clf_alg(choose_alg,X_train,y_train)
    model_performance(model_alg,X_train,y_train)
    model_performance(model_alg,X_test,y_test)
    return model_alg

def model_generate(data):
    data = load_train_set(data)
    data_processed = preprocess_data(data)
    trained_model = model_training(data_processed)
    dump(trained_model,'developed.joblib')
    return trained_model

developed_model=model_generate("training_data.csv")