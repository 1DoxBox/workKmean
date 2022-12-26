import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


dataset = pd.read_csv('csv/CAR DETAILS FROM CAR DEKHO.csv')
dataset = dataset.iloc[:,[0,4]]

vectorizer = TfidfVectorizer(sublinear_tf= True, min_df=10, norm='l2', ngram_range=(1, 2), stop_words='english')
X_train_vc = vectorizer.fit_transform(dataset["name"])
pd.DataFrame(X_train_vc.toarray(), columns=vectorizer.get_feature_names_out()).head()

k_clusters = 10
score = []
for i in range(1,k_clusters+1):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=3000,n_init=5,random_state=0)
    kmeans.fit(X_train_vc)
    score.append(kmeans.i)