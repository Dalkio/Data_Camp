from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.decomposition import PCA, KernelPCA                                        
from sklearn.pipeline import Pipeline                                            
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import numpy as np   
  
concentration = {'A': [300, 400, 600, 800, 1000, 1400, 1600, 2000, 5000, 10000],
                 'B': [500, 1000, 1500, 2000, 4000, 5000, 7000, 10000, 20000, 25000],
                 'Q': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                 'R': [400, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000, 10000]
                }
                                              
class Regressor(BaseEstimator):                          
    def __init__(self):                                                          
        self.n_components = 9
        self.n_estimators = 1500
        self.learning_rate = 0.2                                            
        self.list_molecule = ['A', 'B', 'Q', 'R']                      
        self.dict_reg = {}                                                       
        for mol in self.list_molecule:                                           
            self.dict_reg[mol] = Pipeline([
                ('kpca', KernelPCA(n_components=self.n_components, fit_inverse_transform=True, eigen_solver='arpack')),                
                ('reg', GradientBoostingRegressor(loss='huber', learning_rate=0.02, n_estimators=self.n_estimators, subsample=0.35, random_state=42, alpha=0.95))
            ])                                                                   
                                                                                 
    def fit(self, X, y):                                                         
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol]                                                  
            y_mol = y[ind_mol].astype(float)                                
            self.dict_reg[mol].fit(XX_mol, np.log(y_mol))                        
                                                                                 
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])                                            
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol].astype(float)                                    
            y_pred[ind_mol] = np.exp(self.dict_reg[mol].predict(XX_mol))
                        
            vector_concentration = np.zeros(y_pred[ind_mol].shape) 
            for j, predict_conc in enumerate(y_pred[ind_mol]):
                indice = np.argmin(np.abs(np.ones(len(concentration[mol]))*predict_conc - np.asarray(concentration[mol])))
                vector_concentration[j] = concentration[mol][indice]

            y_pred[ind_mol] = vector_concentration
            
        return y_pred                                                                            