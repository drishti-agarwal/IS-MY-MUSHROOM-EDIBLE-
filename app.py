import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title('IS MY MUSHROOM EDIBLE ?')
    st.sidebar.title('MENU')
    st.markdown("**_ABOUT_**")
    st.markdown("All mushrooms are fungi and they" 
    "produce spores, similar to pollen or seeds, which" 
    "allows them to spread or travel by the wind." 
    "The rest of the mushroom then matures, typically living" 
    "in soil or wood.There are many different types of mushrooms,"
    "some of which are edible including well-known species such as"
    "button, oyster, porcini and chanterelles. There are," 
    "however, many species that are not edible and can in" 
    "fact cause stomach pains or vomiting if eaten, and in"
    "some cases could be fatal, such as the common death cap mushroom. ")
    
    st.sidebar.markdown("Apply Machine Learning models to know the statistic...")
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("./mushroom.csv")
        label =LabelEncoder()
        for i in data.columns:
            data[i] = label.fit_transform(data[i])
        return data 

    df = load_data()

    if st.sidebar.checkbox("Dataset",False):
        st.subheader("The Data for Classification")
        st.write(df) 
    @st.cache(persist=True)
    def split(df):
        X = df.drop(columns=['type'])
        Y = df.type
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
        return X_train,X_test,Y_train,Y_test

    X_train,X_test,Y_train,Y_test = split(df)

    
    
    def plot_metrices(metrices):
        if 'Confusion Matrix' in metrices:
            st.subheader('CONFUSION MATRIX')
            plot_confusion_matrix(model,X_test,Y_test,display_labels=["poisonous","edible"])
            st.pyplot()
        if 'ROC-Curve' in metrices:
            st.subheader('ROC-CURVE')
            plot_roc_curve(model,X_test,Y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in metrices:
            st.subheader('PRECISION-RECALL CURVE')
            plot_precision_recall_curve(model,X_test,Y_test)
            st.pyplot()
        
    st.sidebar.subheader("PREDICITON ALGORITHM")
    algo = st.sidebar.selectbox("Classifiers",('Support Vector Machine(SVM)','Logistic Regression','Random Forest Classifier'))
    
    if algo == 'Support Vector Machine(SVM)':
        st.sidebar.subheader("Enter the Hyperparameters")
        C = st.sidebar.number_input("Regularization Parameter",0.01,10.0,step=0.01,key='C')
        kernel = st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
        gamma = st.sidebar.radio("Kernel Coeeficient",("auto","scale"),key='gamma')
        metrices = st.sidebar.multiselect("Choose the plots",('Confusion Matrix','ROC-Curve','Precision-Recall Curve'))
        if st.sidebar.button("SHOW RESULTS",key='result'):
            model = SVC(C=C,kernel=kernel,gamma=gamma,random_state=0)
            model.fit(X_train,Y_train)
            Y_pred = model.predict(X_test)
            st.write("ACCURACY:",model.score(X_test,Y_test).round(2))
            st.write("PRECISION:",precision_score(Y_test,Y_pred,labels=["poisonous","edible"]).round(2))
            st.write("RECALL:",recall_score(Y_test,Y_pred,labels=["poisonous","edible"]).round(2))
            plot_metrices(metrices)

    if algo == 'Logistic Regression':
        st.sidebar.subheader("Enter the Hyperparameters")
        C = st.sidebar.number_input("Regularization Parameter",0.01,10.0,step=0.01,key='C')
        max_iter = st.sidebar.slider("Maximum Iterations",100,500,key='max_iter')
        metrices = st.sidebar.multiselect("Choose the plots",('Confusion Matrix','ROC-Curve','Precision-Recall Curve'))
        if st.sidebar.button("SHOW RESULTS",key='result'):
            model = LogisticRegression(C=C,random_state=0,max_iter=max_iter)
            model.fit(X_train,Y_train)
            Y_pred = model.predict(X_test)
            st.write("ACCURACY:",model.score(X_test,Y_test).round(2))
            st.write("PRECISION:",precision_score(Y_test,Y_pred,labels=["poisonous","edible"]).round(2))
            st.write("RECALL:",recall_score(Y_test,Y_pred,labels=["poisonous","edible"]).round(2))
            plot_metrices(metrices)

    if algo == 'Random Forest Classifier':
        st.sidebar.subheader("Enter the Hyperparameters")
        n_estimators = st.sidebar.number_input("Maximum Estimators",100,500,step=10,key='n_estimator')
        max_depth = st.sidebar.number_input("Maximum Depth of Classifier",1,20,step=1,key='max_depth')
        bootstrap = st.sidebar.radio("Bootsrap",('true','false'),key='bootstrap')
        metrices = st.sidebar.multiselect("Choose the plots",('Confusion Matrix','ROC-Curve','Precision-Recall Curve'))
        if st.sidebar.button("SHOW RESULTS",key='result'):
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            model.fit(X_train,Y_train)
            Y_pred = model.predict(X_test)
            st.write("ACCURACY:",model.score(X_test,Y_test).round(2))
            st.write("PRECISION:",precision_score(Y_test,Y_pred,labels=["poisonous","edible"]).round(2))
            st.write("RECALL:",recall_score(Y_test,Y_pred,labels=["poisonous","edible"]).round(2))
            plot_metrices(metrices)

    
          

    
    
if __name__ == "__main__":
    main()    







   











  