# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from pandas.plotting import scatter_matrix
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def load_data():
    X_train = np.load(open('../res/X_train.npy','rb'))
    X_test =np.load(open('../res/X_test.npy','rb'))
    y_train = np.load(open('../res/y_train.npy','rb'))
    y_test = np.load(open('../res/y_test.npy','rb'))
    return X_train,X_test,y_train,y_test



def describe_df(X):
    print(X.describe())




# =============================================================================
# To visualise data, you need to pass training data only as the assumption holds that test set is unknown data and obviously,you cant not make decision based on unseen data :-p

#Remember to concatenate training features and labels if you want to check that scatterplots which I would prefer.You are free to explore labels to labels, features to features ,etc scatterplots as you want by passing arguments
#============================================================================


def plot_corr(data, size=11):
    corr = data.corr()
    fig, ax = subplots(figsize=(size, size))
    set_cmap("YlOrRd")
    ax.matshow(corr)
    xticks(range(len(corr.columns)), corr.columns, rotation=90)
    print(len(corr.columns))
    yticks(range(len(corr.columns)), corr.columns)
    fig.savefig("../out/corr.jpg")
    return ax



def feature_selection_pca(X_train,k):
    pca = PCA(n_components=k)
    X = pca.fit_transform(X_train)
    pickle.dump(pca,open('../res/pca.pkl','wb'))
    print(pca.explained_variance_ratio_) 
    return X





def percentile_k_features(X,y, k):
    model = f_regression
    skb = SelectPercentile(model, k)
    predictors = skb.fit_transform(X,y)
    scores = list(skb.scores_)
    
    top_k_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:predictors.shape[1]]
    top_k_predictors = [X.columns[i] for i in top_k_index]
    
    X_indices = np.arange(X.shape[1])
    scores = -np.log10(skb.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)',  color='darkorange', edgecolor='black')
    plt.xticks(X_indices+np.ones(X_indices.shape), X.columns,rotation=90 )
    plt.savefig("../out/select_percentile_graph.png")
    return top_k_predictors

def feature_selection(X,y,k):
    feat = percentile_k_features(X,y, k)
    return feat

def ohe_encode(X,category_index):
    ohe = OneHotEncoder(categorical_features=category_index,sparse=False)
    X_transform = ohe.fit_transform(X)
    pickle.dump(ohe,open('../res/ohe.pkl','wb'))
    return X_transform



