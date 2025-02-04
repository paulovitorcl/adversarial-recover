# classifier_poly.py
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

class PolynomialClassifier:
    def __init__(self, degree=4):
        self.degree = degree
        self.poly = None
        self.model = None
    
    def fit(self, X, y):
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_poly, y)
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)
    
    def predict_proba(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict_proba(X_poly)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            y_prob = self.predict_proba(X)[:,1]
            auc = roc_auc_score(y, y_prob)
            return acc, auc
        else:
            return acc, None
