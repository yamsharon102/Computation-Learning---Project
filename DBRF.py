import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.ensemble import RandomForestClassifier
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import time

class DBRF:
    
    def __init__(self, n=50, n_estimators=50, criterion='gini'):
        self.n = round(n)
        self.n_estimators = round(n_estimators)
        self.criterion = criterion
        self.F_list = []
        self.H_list = []
        
    def get_params(self, deep=True):
        new_n = copy.copy(self.n)
        new_est = copy.copy(self.n_estimators)
        new_crit = copy.copy(self.criterion)
        return {'n' : new_n, 'n_estimators' : new_est, 'criterion' : new_crit}
        
    def HEM(self, df, y, classifier):
        leafs_scores = dict()
        df_values = np.array(df.values, dtype=np.float32)
        for i, val in (enumerate(df_values)):
            curr_leafs = classifier.apply(np.array([val], dtype=np.float32))
            for j, estimator in enumerate(classifier.estimators_):
                correct = int(estimator.predict(np.array([val], dtype=np.float32))[0] == y.iloc[i])
                if curr_leafs[0][j] not in leafs_scores:
                    leafs_scores[curr_leafs[0][j]] = [0, 0]
                leafs_scores[curr_leafs[0][j]][0] += correct
                leafs_scores[curr_leafs[0][j]][1] += 1

        leaf_supp = dict()
        leaf_conf = dict()
        for leaf in leafs_scores:
            leaf_supp[leaf] = leafs_scores[leaf][1] / len(df)
            leaf_conf[leaf] = leafs_scores[leaf][0] / leafs_scores[leaf][1]

        for leaf in leafs_scores:
            leafs_scores[leaf] = (2*leaf_supp[leaf]*leaf_conf[leaf]) / (leaf_supp[leaf]+leaf_conf[leaf])

        return leafs_scores
    
    def split_data(self, df, indexes, classifier, H_i):
        sigma = np.average(list(H_i.values()))
        df_values = np.array(df.values, dtype=np.float32)
        D_e, D_h = [], []
        for i, val in (enumerate(df_values)):
            curr_leafs = classifier.apply(np.array([val], dtype=np.float32))
            for leaf in curr_leafs[0]:
                if H_i[leaf] <= sigma:
                    D_h.append(indexes[i])
                    break
            D_e.append(indexes[i])
        return D_e, D_h
    
    def score(self, X, y):
        pred_y = self.predict(X)
        return accuracy_score(y, pred_y)
    
    def set_params(self, criterion, n_estimators, n):
        self.n = copy.copy(round(n))
        self.n_estimators = copy.copy(round(n_estimators))
        self.criterion = copy.copy(criterion)
        return self
        
    def get_params(self, deep=True):
        return {'n' : self.n, 'n_estimators' : self.n_estimators, 'criterion' : self.criterion}

    
    def fit(self, D_d_x, D_d_y):
        D_d = D_d_x
        D_d['my_new_class'] = D_d_y
        D = copy.deepcopy(D_d) # 2.
        for i in range(self.n): # 3.    
            
            indexes = list(range(len(D)))
            
            # 4. RF
            F_i = RandomForestClassifier(random_state=0, n_estimators=self.n_estimators, criterion=self.criterion)
            X = D[D.columns[:-1]]
            y = D[D.columns[-1]]
            
            F_i.fit(X,y)

            # 5. HEM
            H_i = self.HEM(X, y, F_i)

            # 6. Split
            D_e, D_h = self.split_data(X, indexes, F_i, H_i)
            

            # 7. D <- D_h

            indexes = D_h
            D = X.iloc[indexes]
            D['my_new_class'] = y.iloc[indexes]

            # 8. + 9.
            self.F_list.append(F_i)
            self.H_list.append(H_i)
            
    def predict(self, D_t):
        O = np.ones(len(D_t)) * -1
        D = copy.deepcopy(D_t)
        for i in range(self.n):
            try:
                indexes = list(range(len(D)))

                F_i = self.F_list[i]
                H_i = self.H_list[i]

                D_e, D_h = self.split_data(D, indexes, F_i, H_i)

                preds = F_i.predict(D.iloc[D_e,:])
                for i, index in enumerate(D_e):
                    O[index] = preds[i]

                D = copy.deepcopy(D.iloc[D_h,:])
                indexes = D_h

            except Exception as e:
                raise e
                return O
        
        preds = F_i.predict(D)
        for i, index in enumerate(D_h):
            O[index] = preds[i]

        return O
    
    
    def predict_proba(self, D_t):
        O = [[]]*len(D_t)
        D = copy.deepcopy(D_t)
        for i in range(self.n):
            try:
                indexes = list(range(len(D)))

                F_i = self.F_list[i]
                H_i = self.H_list[i]

                D_e, D_h = self.split_data(D, indexes, F_i, H_i)

                preds = F_i.predict_proba(D.iloc[D_e,:])
                for i, index in enumerate(D_e):
                    O[index] = preds[i]

                D = copy.deepcopy(D.iloc[D_h,:])
                indexes = D_h

            except:
                return O
        
        preds = F_i.predict_proba(D)
        for i, index in enumerate(D_h):
            O[index] = preds[i]

        return np.array(O)


