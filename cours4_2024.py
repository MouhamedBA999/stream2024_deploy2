import streamlit as st
from  sklearn import datasets 
import numpy as np 
import seaborn as sns

##Modelss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA 
import matplotlib.pyplot as  plt 

##creation de Texte
st.header('''
Explore different Classifier 
Which on the best ?
''')

##mettre la selection à gauche
dataset_name=st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine dataset'))
st.write(dataset_name)

##Selectionner le classifier
classifier_name=st.sidebar.selectbox('Select le Classifieur', ('KNN', 'SVM', 'Random Forest'))

###Selection de dataset

def get_dataset(dataset_name):
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='Breast Cancer':
        data=datasets.load_breast_cancer()
    else :
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y    
X,y=get_dataset(dataset_name)
st.write('Overvier of data')
st.dataframe(X)

st.write('Shape of dataset', X.shape)
st.write('Number of classe : ', len(np.unique(y)))

##Ajouter les paramètres de notre modèle 
def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=='KNN':
        K=st.sidebar.slider('K', 1, 15)
        params['K']=K
    elif clf_name=='SVM':
        C=st.sidebar.slider("C",0.01, 10.0)
        params['C']=C
    else:
        max_depth=st.sidebar.slider('max_depth', 2, 15)
        n_estimators=st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth']=max_depth
        params['n_estimators']=n_estimators
    return params

params=add_parameter_ui(classifier_name)


##Mise en place des modèles 

def get_classifier(clf_name, params):
    if clf_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name=='SVM':
        clf=SVC(C=params['C'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    return clf

clf=get_classifier(classifier_name, params)

##Train test Split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_test, y_pred)
st.write(f'classifier={classifier_name}')
st.write(f'accuracy={acc}')

##Matrice de confusion 
conf_matrix=confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

pca=PCA(2)
x_projete=pca.fit_transform(X_test)

##Afficher la variance expliquée par ACP : ou bien le taux d'information
##apporté par les nouvelles composantes princpiales
st.write( 'la vairance expliquée est :',pca.explained_variance_ratio_)

##Graphique
x1=x_projete[:,0]
x2=x_projete[:,1]

fig, (ax1, ax2)=plt.subplots(1,2)

ax1.scatter(x1,x2, c=y_test, alpha=0.8, cmap='viridis')
ax1.set_title('label')

ax2.scatter(x1,x2, c=y_pred, alpha=0.8, cmap='viridis')
ax2.set_title('label')

st.pyplot(fig)































