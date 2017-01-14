import numpy as np

labels = np.array(['A', 'B', 'Q', 'R'])

class FeatureExtractorReg():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        
        XX_max = np.max(XX, axis=1)
        
        XX -= np.mean(XX, axis=1)[:, None]
        XX /= np.sqrt(np.sum(XX ** 2, axis=1))[:, None]
        
        XX = np.c_[XX, XX_max, X_df[labels].values]
        
        return XX

