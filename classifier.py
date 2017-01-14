from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV

class Classifier(BaseEstimator):
    def __init__(self):
        
        self.n_components = 10 # Arack as eigen_solver then, because n_components much smaller than number_samples
        self.n_estimators = 300
        self.clf = Pipeline([
            ('kpca', KernelPCA(n_components=self.n_components, fit_inverse_transform=True, eigen_solver='arpack')),
            ('svc', SVC(C=1e6, probability=True))
        ])

    def fit(self, X, y):
        
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        prediction = self.clf.predict_proba(X)
        prediction[:,1] *= 1.42 # Error on B predominant
        return prediction