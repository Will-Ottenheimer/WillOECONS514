#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Flow of the code
# Long format data
# Calculate probability of each alternative
# P is calculated by creating design matrix X... 
# then create vector V of desired index values for each alternative for each individual
# you do this by finding the dot product of X and Beta hat. 
# Indirect value = V = X*Bhat = P = exp(s) / lamda (exp(s))
#lambda is mapping matrix
# once we find a theta hat we can use optimization package to find gradient of estimate


#"C:\Users\otten\Desktop\econs514\assignment 1.txt" #
# "C:\Users\otten\Desktop\assignment 1.txt"  #


# In[ ]:


import os
import numpy as np
import pandas as pd
import scipy.optimize as opt


# In[ ]:


### 1. Develop a binary choice model on commuters’ transponder acquisition decision.
### 2. Develop a multinomial choice model on commuters’ joint occupancy/route choices.
### 3. Develop a nested-logit model in which commuters first choose to acquire a transponder; then 
### choose car occupancy; then choose the road.
### 4. It is reasonable to postulate that commuters are different in their preferences for travel time 
###   and money costs. Try to investigate the heterogeneity among commuters based on the model 
###   developed in step 2. 


# In[2]:


# Question 1 binomial choice

class ChoiceModels(object):
    
    '''
    This class defines methods that will be used later in speficying and estimating choice models.
    '''
   
    def load_data(self, path, file):
        df = pd.read_csv("C:\\Users\\otten\\Desktop\\assignment 1.txt" , sep='\s+', header=0)
        df['cons'] = 1.
        return df
    
    def expand_data(self, df, n):
        '''
        Parameters
        ----------
        df : a pandas data frame
            
        n : Integer
            Number of times to expand the data
        Returns
        -------
        An expanded pandas data frame with a panel structure
        '''
        df['Alt'] = [[str(i) for i in range(n)] for _ in range(len(df))]
        return df.explode('Alt')
    
    def create_choice_attributes(self, df, config):
        '''
        This method creates a panel structure of data to estimate the multinomial
        choice model speficied in the configuration file (config-- a json format file)
        '''
        # create dependent variable
        y_namelist = list(config['Alternatives']['0'].keys())
        df['choice'] = list(zip(*[df[v] for v in y_namelist]))
        df = self.expand_data(df, len(config['Alternatives']))
   
        df['y'] = 0.
        for k,v in config['Alternatives'].items():
            label = tuple(v.values())
            df.loc[(df["Alt"]==k) & (df['choice']==label), 'y'] = 1
        
        # create alternative specific attributes
        dic = config['Attributes']
        for var,info in dic.items():
            df[var] = 0
            for alt, w in info['rule'].items():
                df['junk'] = 0
                df.loc[(df['Alt'] == alt), 'junk'] = 1
                df[var] = df[var] + w * df[info['variable']] * df['junk'] 
        df = df.drop("junk", axis='columns')
        
        # creat interactions
        df, xz_list = self.create_interactions(df, config['Interactions']) 
        x_list = list(config['Attributes'].keys()) + xz_list
        return {'data': df, "var_names": x_list}
    
    def create_interactions(self, df, interact_list):
        '''
        Parameters
        ----------
        df : pandas data frame
            
        interact_list : a List
            The list contains pairs of variable names as tuples
        Returns
        -------
        df : pandas data frame after adding interactions
            
        xz_list : A list of created interactions
        '''
        xz_list = []
        for item in interact_list:
            vname = item[0] + "_" + item[1]
            df[vname] = df[item[0]] * df[item[1]]
            xz_list.append(vname)
        return df, xz_list 
        
        
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
        return {'log_likelihood':-1*v.fun, "Coefficients": v.x, "Var_Cov": v.hess_inv}

class BinaryLogit(ChoiceModels):
    '''
    This class is to estimate a binary logit nodel by MLE.  
    '''
    def __init__(self, path, file, yname, x=None, z=None, interactions=None):
        df = super().load_data(path, file)
        if x is None:
            x = []
        if z is None:
            z = []
        if interactions is None:
            xz = []
            self.df = df
        else:
            self.df, xz = super().create_interactions(df, interactions)
            
        self.X_list = ['cons'] + x + z + xz
        self.Xmat = self.df[self.X_list].to_numpy()
        self.y = self.df[yname].to_numpy()
        
    def log_likelihood(self, para):
        '''
        Parameters
        ----------
        para : array
            a 1-D array with the shape(k,), where k is the number of model parameters.
        Returns
        -------
        res : scalar
            log-likelihood value
        '''
        xb = np.matmul(self.Xmat, para)
        xb = np.exp(xb)
        xb = xb / (1+xb)
        return (-1/len(xb)) * np.sum(self.y * np.log(xb) + (1-self.y) * np.log(1 - xb))
   
    def estimation(self, para):
        '''
        Parameters
        ----------
        para : array
            a 1-D array with the shape(k,), where k is the number of model parameters.
        Returns
        -------
        A dictionary of estimation results
        '''
        return super().optimization(self.log_likelihood, para)
    
if __name__ == '__main__':
    p = r"C:\\Users\\otten\\Desktop\\"
    f = "assignment 1.txt"
    ## estimating binary models
    x = ['toll', 'median']
    z = ['female', 'age3050', 'householdsize']
    interactions = [('toll', 'high_income'), ('toll', 'med_income')]
    trans= BinaryLogit(p,f,"trans", x=x,z=z,interactions=interactions)
    bini = np.zeros(len(trans.X_list))
    res_binary = trans.estimation(bini)
    print(res_binary)
   
    


# In[3]:


# Question 2 Multinomial choice

class MultinomialLogit(ChoiceModels):

    # Specify model here    
    model_config = {"Alternatives":
                    {"0": {"trans": 1, "occupanc": 1, "route": 1},
                     "1": {"trans": 1, "occupanc": 1, "route": 0},
                     "2": {"trans": 1, "occupanc": 2, "route": 1},
                     "3": {"trans": 1, "occupanc": 2, "route": 0},
                     "4": {"trans": 1, "occupanc": 3, "route": 1},
                     "5": {"trans": 1, "occupanc": 3, "route": 0},
                     "6": {"trans": 0, "occupanc": 1, "route": 0},
                     "7": {"trans": 0, "occupanc": 2, "route": 0},
                     "8": {"trans": 0, "occupanc": 3, "route": 0}},
                    "Nests": {"0":{"0": ["0", "1"], "1": ["2", "3"], 
                                   "2": ["4", "5"]},"1":["6", "7", "8"]},
    
                    "Attributes":{'trans_dummy':{'variable': 'cons', 
                                                 'rule':{"0":1,"1":1,
                                                         "2":1,"3":1,"4":1,"5":1}},
                                  'express_dummy':{'variable':'cons', 
                                                   'rule':{"0":1,"2":1,"4":1}},
                                  'hov2_dummy':{'variable':'cons', 
                                               'rule':{"2":1,"3":1,"7":1}},
                                  "hov3_dummy":{'variable':'cons', 
                                                'rule':{"4":1,"5":1,"8":1}},
                                  "price":{"variable": 'toll', 
                                           "rule": {"0":1,"2":1/2,"4":1/6}},
                                  "time": {"variable":"median", 
                                           "rule":{"0":1,"2":1,"4":1}}},
                    "Interactions":[('price', "high_income"), ('price', "med_income"),
                                    ("hov2_dummy", "householdsize"),
                                    ("hov3_dummy", "householdsize")],
                    "Mixedlogit":['price', 'time', 'trans_dummy', 'hov2_dummy', 'hov3_dummy']}
    
    
    def __init__(self, path, file):
        df = super().load_data(path, file)
        res = super().create_choice_attributes(df, MultinomialLogit.model_config)
        self.df = res['data']
        self.X_list = res['var_names']
        self.y = self.df['y'].to_numpy()
        self.Xmat = self.df[self.X_list].to_numpy()
               
    def mnl_log_likelihood(self, para):
        '''
        This method defines the data log-likelihood from a Multinomial Logit.
        '''
        df = self.df.copy()
        xb = np.matmul(self.Xmat, para)
        xb = np.exp(xb)
        df['xb'] = xb.tolist()
        # group sum
        df['xbsum'] = df.groupby(['id'])["xb"].transform(lambda x: x.sum())
        df['log_likelihood'] = df['y']*np.log(df['xb'] / df['xbsum'])
        return (-1/len(df))* np.sum(df['log_likelihood'])
 
    def estimation(self, para):
        '''
        Parameters
        ----------
        para : array
            a 1-D array with the shape(k,), where k is the number of model parameters.
        Returns
        -------
        A dictionary of estimation results
        '''
        return super().optimization(self.mnl_log_likelihood, para)
    
if __name__ == '__main__':
    p = r"C:\\Users\\otten\\Desktop\\"
    f = "assignment 1.txt"
    ## estimating binary models
    x = ['toll', 'median']
    z = ['female', 'age3050', 'distance', 'householdsize']
    interactions = [('toll', 'high_income'), ('toll', 'med_income')]
    route = BinaryLogit(p, f, "route", x=x, z=z, interactions=interactions)

    
    ## estimating a MNL model
    mnl = MultinomialLogit(p, f)
    bini = np.zeros(len(mnl.X_list))
    res_mnl = mnl.estimation(bini)
print(res_mnl)


# In[14]:


# Question 3 Nested Logit choice

class MultinomialLogit(ChoiceModels):

    # Specify model here    
    model_config = {"Alternatives":
                    {"0": {"trans": 1, "occupanc": 1, "route": 1},
                     "1": {"trans": 1, "occupanc": 1, "route": 0},
                     "2": {"trans": 1, "occupanc": 2, "route": 1},
                     "3": {"trans": 1, "occupanc": 2, "route": 0},
                     "4": {"trans": 1, "occupanc": 3, "route": 1},
                     "5": {"trans": 1, "occupanc": 3, "route": 0},
                     "6": {"trans": 0, "occupanc": 1, "route": 0},
                     "7": {"trans": 0, "occupanc": 2, "route": 0},
                     "8": {"trans": 0, "occupanc": 3, "route": 0}},
                    "Nests": {"0":{"0": ["0", "1"], "1": ["2", "3"], 
                                   "2": ["4", "5"]},"1":["6", "7", "8"]},
    
                    "Attributes":{'trans_dummy':{'variable': 'cons', 
                                                 'rule':{"0":1,"1":1,
                                                         "2":1,"3":1,"4":1,"5":1}},
                                  'express_dummy':{'variable':'cons', 
                                                   'rule':{"0":1,"2":1,"4":1}},
                                  'hov2_dummy':{'variable':'cons', 
                                               'rule':{"2":1,"3":1,"7":1}},
                                  "hov3_dummy":{'variable':'cons', 
                                                'rule':{"4":1,"5":1,"8":1}},
                                  "price":{"variable": 'toll', 
                                           "rule": {"0":1,"2":1/2,"4":1/6}},
                                  "time": {"variable":"median", 
                                           "rule":{"0":1,"2":1,"4":1}}},
                    "Interactions":[('price', "high_income"), ('price', "med_income"),
                                    ("hov2_dummy", "householdsize"),
                                    ("hov3_dummy", "householdsize")],
                    "Nestedlogit": {"structure": {"0": ["0", "1", "2"], "1": ["6", "7", "8"]}, 
                                    "variables": ['price', 'time', 'trans_dummy', 
                                                  'hov2_dummy', 'hov3_dummy']}
                   }
        
    def __init__(self, path, file):
        df = super().load_data(path, file)
        res = super().create_choice_attributes(df, MultinomialLogit.model_config)
        self.df = res['data']
        self.X_list = res['var_names']
        self.y = self.df['y'].to_numpy()
        self.Xmat = self.df[self.X_list].to_numpy()
               
    def mnl_log_likelihood(self, para):
        '''
        This method defines the data log-likelihood from a Multinomial Logit.
        '''
        df = self.df.copy()
        xb = np.matmul(self.Xmat, para)
        xb = np.exp(xb)
        df['xb'] = xb.tolist()
        # group sum
        df['xbsum'] = df.groupby(['id'])["xb"].transform(lambda x: x.sum())
        df['log_likelihood'] = df['y']*np.log(df['xb'] / df['xbsum'])
        return (-1/len(df))* np.sum(df['log_likelihood'])
    
    def estimation(self, para):
        '''
        Parameters
        ----------
        para : array
            a 1-D array with the shape(k,), where k is the number of model parameters.
        Returns
        -------
        A dictionary of estimation results
        '''
        return super().optimization(self.mnl_log_likelihood, para)
    
    def mnl_log_likelihood(self, para):
        df = self.df.copy()
        xb = np.matmul(self.Xmat, para)
        xb = np.exp(xb/0.5)
        df['xb'] = xb.tolist()
    # group sum
        df['xbsum'] = df.groupby(['id'])["xb"].transform(lambda x: x.sum())
        df['nom_1'] = df['xbsum']**(-0.5)
        df['nom_2'] = df['xb']*df['nom_1']
        df['denom_1'] = df['xbsum']**0.5
        df['denom_2'] = np.sum(df['denom_1'])
        df['log_likelihood'] = df['y']*np.log(df['nom_2']/df['denom_2'])
        return (-1/len(df))*np.sum(df['log_likelihood'])
    def estimation(self, para):
        return super().optimization(self.mnl_log_likelihood, para)
if __name__ == '__main__':
    p = r"C:\\Users\\otten\\Desktop\\"
    f = "assignment 1.txt"

    mnl = MultinomialLogit(p,f)
    bini = np.zeros(len(mnl.X_list))
    res_mnl = mnl.estimation(bini)
print(" ")
print("Question 3: Y=Trans,Occu,Route")
print(" ")
print(res_mnl)


# In[ ]:


# Question 4 are commuters different in their preferences for travel time and money costs? Estimate Heterogeneity among commuters in multinomial

#Redo the model configuration with randomization for the price and time price

