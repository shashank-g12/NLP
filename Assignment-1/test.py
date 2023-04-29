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

model_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]


def evaluation(y_true, y_pred):
	f1_micro = f1_score(y_true, y_pred, average = 'micro')
	f1_macro = f1_score(y_true, y_pred, average = 'macro')
	print(f"f1_micro: {f1_micro}\nf1_macro: {f1_macro}\nAverage: {(f1_micro+f1_macro)/2}")

def preprocessing(X):
	for i in range(len(X)):
		X[i] = X[i].translate(str.maketrans('','', string.punctuation)).lower()
	return X

def main():
	clf, vectorizer = joblib.load(model_path)
	test_df = pd.read_csv(test_path, header=None).dropna()
	X = np.array(test_df[0])
	X = preprocessing(X)
	X_test = vectorizer.transform(X)
	out_df = pd.DataFrame(clf.predict(X_test))
	out_df.to_csv(output_path, header=False, index=False)
	evaluation(np.array(test_df[1]), np.array(out_df[0]))

if __name__ == '__main__':
	main()

