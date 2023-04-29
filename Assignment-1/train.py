import sys
import os
import numpy as np
import pandas as pd
import sklearn
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train_path = sys.argv[1]
output_path = sys.argv[2]

def evaluation(y_true, y_pred):
	f1_micro = f1_score(y_true, y_pred, average = 'micro')
	f1_macro = f1_score(y_true, y_pred, average = 'macro')
	print(f"f1_micro: {f1_micro}\nf1_macro: {f1_macro}\nAverage: {(f1_micro+f1_macro)/2}")

def preprocessing(X):
	for i in range(len(X)):
		X[i] = X[i].translate(str.maketrans('','', string.punctuation)).lower()
	return X

def main():
	train_df = pd.read_csv(train_path, header=None).dropna()
	X, y = np.array(train_df[0]), np.array(train_df[1])
	X = preprocessing(X)
	vectorizer = CountVectorizer(max_features=4000,ngram_range=(1,2))
	X_train = vectorizer.fit_transform(X)
	clf = LogisticRegression(C=0.5, max_iter=1000).fit(X_train, y)
	joblib.dump((clf,vectorizer), output_path)
	evaluation(y, clf.predict(X_train))

if __name__ == '__main__':
	main()
