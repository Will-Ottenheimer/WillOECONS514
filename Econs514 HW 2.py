#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
import numpy as np
import scipy.io
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

class BLP(object):
    '''
    Jia Yan
    02/21/2023
    '''
    
    def __init__(self, path, file, ndraws=500, tol_fp=1e-12):
        self.ndraws = ndraws
        self.tol_fp = tol_fp
        data = scipy.io.loadmat(os.path.join("C:\\Users\\otten\\Desktop\\econs514\\BLP_data.mat"))
        v_list = ['outshr', 'const', 'mpd', 'air', 'space', 'hpwt', 'price', 'trend']
        self.nmarkets = data['trend'].max() + 1
        self.df = data['share']
        for item in v_list:
            self.df = np.concatenate([self.df, data[item]], axis=1)
        self.df = pd.DataFrame(self.df, columns = ['share'] + v_list)
        
        self.attributes = ['const', 'mpd', 'air', 'space', 'hpwt', 'price']
        self.Xmat = self.df[self.attributes].to_numpy()
        self.attributes_random = ['const', 'mpd', 'air', 'space', 'hpwt']
        
        '''
        Take standard normal draws
        '''
        self.draws = np.random.randn(self.nmarkets, self.ndraws, len(self.attributes_random))    
        
        '''
        creat instruments for price: sum of attributes of rival products
        '''
        z_list = ['mpd', 'air', 'space', 'hpwt']
        self.IV_list = ['const', 'mpd', 'air', 'space', 'hpwt'] # the first part of IV are exogenous regressors
        for var in z_list:
            name = var + "_" + "z"
            self.IV_list.append(name)
            self.df[name] = self.df.groupby(['trend'])[var].transform(lambda x: x.sum())
            self.df[name] = self.df[name] - self.df[var]
        self.Zmat = self.df[self.IV_list].to_numpy()
        self.weight_mat = np.linalg.inv(np.matmul(np.transpose(self.Zmat), self.Zmat)) # weighting matrix in GMM estimation
        pz = np.matmul(self.Zmat, self.weight_mat)
        pz = np.matmul(pz, np.transpose(self.Zmat))
        self.project_mat = np.matmul(np.transpose(self.Xmat), pz)
        self.project_mat = np.matmul(self.project_mat, self.Xmat)
        self.project_mat = np.linalg.inv(self.project_mat)
        self.project_mat = np.matmul(self.project_mat, np.transpose(self.Xmat))
        self.project_mat = np.matmul(self.project_mat, pz)
        
    def ols(self):
        '''
        replicate the first column of table 3
        '''
        y = np.log(self.df['share']/self.df['outshr'])
        #b = np.matmul(np.transpose(self.Xmat), self.Xmat)
        #b = np.linalg.inv(b)
        #b = np.matmul(b, np.transpose(self.Xmat))
        #return np.matmul(b, y)
        return sm.OLS(y, self.Xmat).fit()
        
    def iv(self):
        '''
        replicate the second column of table 3
        '''
        y = np.log(self.df['share']/self.df['outshr'])
        #return np.matmul(self.project_mat, y)
        return IV2SLS(y, self.Xmat, self.Zmat).fit()
    
    def market_share(self, mid, delta, xv):
        draws = self.draws[mid]
        s = np.zeros(len(delta))
        for r in range(self.ndraws):
            w = draws[r]
            v = np.exp(delta + (w * xv).sum(axis=1))
            s = s + (v / (1 + np.sum(v)))
        
        return (1/self.ndraws) * s 
    
    def fixed_point(self, pack):
        mid = pack['mid']
        df = pack['df']
        sigmas = pack['sigmas']
        s0 = df['share'].to_numpy()
        xv = sigmas * df[self.attributes_random].to_numpy()
        check = 1.0
        delta_ini = np.zeros(len(s0))
        while check > self.tol_fp:
            delta_new = delta_ini + (np.log(s0) - np.log(self.market_share(mid, delta_ini, xv)))
            check = abs(np.max(delta_new - delta_ini))
            delta_ini = delta_new
        return delta_new
        
    def GMM_obj(self,sigmas):
        """
        sigmas: an 1_D array with the shape (len(self.attributes_random), ), which contains
        the standard errors of random coefficients
        """
        df = self.df.copy()
        v_list = ['share'] + self.attributes_random
        
        '''
        # step 1: solve mean utility (delta_j) from the fixed-point iteration
        '''
        df_list = [{'mid': int(mid), 'df': d[v_list], 'sigmas': sigmas} for mid, d in df.groupby(['trend'])]
        delta_j = tuple(map(self.fixed_point, df_list))
        delta_j = np.concatenate(delta_j, axis=0) # an array with the shape(2217,)
        
        '''
        step 2: uncover mean part of coefficients (beta_bar) from delta_j, which is equivalent to 
        running an IV estimation using delta_j as the dependent variable
        '''
        beta_bar = np.matmul(self.project_mat, delta_j) 
        
        '''
        step 3: uncover ommited product attributes (xi_j) from delta_j and beta_bar
        '''
        xi_j = delta_j - np.matmul(self.Xmat, beta_bar)
        
        '''
        step 4: interact xi_j with instruments,which include exogenous regressors (veihicles' own
        exogenous attributes) and instruments for price (sum of attributes of competing products)
        '''
        moments = np.matmul(np.transpose(self.Zmat), xi_j) # an array with the shape (m, ), where m is the number of IVs
        
        '''
        step 5: compute the GMM objective function
        '''
        f = np.matmul(moments, self.weight_mat)
        f = np.matmul(f, moments)
        return f
    
    def optimization(self, objfun, para):
        '''
        Parameters
        ----------
        objfun : a user defined objective function of para
            
        para : a 1-D array with the shape (k,), where k is the number of parameters.
        Returns
        -------
        dict
            A dictionary containing estimation results
        '''
        v = opt.minimize(objfun, x0=para, jac=None, method='BFGS', 
                          options={'maxiter': 1000, 'disp': True})  
        return {'obj':v.fun, "Coefficients": v.x}

if __name__ == "__main__":
    blp = BLP("/kaggle/input/blp-data/", "BLP_data.mat")
    pini = np.ones(len(blp.attributes_random)) * 0.2
    x = blp.GMM_obj(pini)
    beta_ols = blp.ols()
    beta_iv = blp.iv()
    print(beta_ols.summary())
    print(beta_iv.summary())
    print(sm.OLS(blp.df['price'], blp.Zmat).fit().summary()) # first-stage regression in IV estimation



# In[7]:





# In[ ]:




