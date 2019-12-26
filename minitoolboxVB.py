# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:19:48 2019

@author: Valmir.Bucaj
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d
#from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
#import pylab
#import graphviz
#import os

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
#from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error,mean_squared_log_error, r2_score, auc, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import defaultdict





class DropImpute(object):
    def __init__(self,data=None):
        self.data=data











class Outliers(object):
    
    def __init__(self, data=None):
        
        self.data=data
    
        
    def remove_outliers(self):
    
        """ 
        All the values below Q1-1.5*IQR and above Q3+1.5*IQR will be removed.
        The function returns two objects:
        (1) The updated dataframe with the outliers removed and
        (2) The list of labels of the subjects removed
        """
        self.labels=[]
        
        for col in self.data.columns[:-1]:
            q1=self.data.describe()[col]['25%']
            q3=self.data.describe()[col]['75%']
            
            iqr=q3-q1
            
            minimum=q1-1.5*iqr
            maximum=q3+1.5*iqr
            
            outliers=[(index, val) for (index,val) in zip(self.data[col].index,self.data[col]) 
                      if val<minimum or val>maximum]
            #print(outliers)
            for (index, val) in outliers:
                self.labels.append(index)
        self.labels=list(set(self.labels))
        self.data.drop(self.labels,inplace=True)
        return self.data, list(set(self.labels))


class FeatureSelection(object):
    
    def __init__(self,data,r2_score=True,mse=False,
                 regressor=LinearRegression(),classifier=LogisticRegression(solver='linear')):
       
       self.data=data   
       self.r2_score=r2_score
       self.mse=mse
       self.regressor=regressor
       self.classifier=classifier
       
    def feature_selection(self,predictors=None,target=None,boxPlot=True,roc=True,
                          length=8, height=6,x_fontsize=10,y_fontsize=10,title=None,title_fontsize=16,
                          xticks_size=12,yticks_size=12,
                         regModel=True,classModel=False,testSize=0.3,split=False, iterations=10):
       
       
       self.metricScore=defaultdict(list)
       self.fpr=defaultdict(list)
       self.tpr=defaultdict(list)
       self.roc_auc=defaultdict(list)
       self.iterations=iterations
       self.split=split
       self.predictors=predictors
       self.target=target
       self.testSize=testSize
       self.regModel=regModel
       self.classModel=classModel
       self.boxPlot=boxPlot
       self.roc=roc
       self.length=length
       self.height=height
       self.x_fontsize=x_fontsize
       self.y_fontsize=y_fontsize
       self.title=title
       self.title_fontsize=title_fontsize
       self.xticks_size=xticks_size
       self.yticks_size=yticks_size
       
       if self.regModel==True and classModel==False:
           model=self.regressor
           
           if self.r2_score==True and self.mse==False:
               
               if self.split==False:
                   
                   for i in range(iterations):
                       X_train,X_test,y_train,y_test=train_test_split(self.data.drop(self.data.columns[-1],axis=1),
                                                                       self.data[self.data.columns[-1]],
                                                                       test_size=self.testSize, 
                                                                       random_state=np.random.randint(3000))
                       
                       
                       ct=0
                       for feat in self.data.columns[:-1]:                           
                                    
                            model.fit(X_train,y_train)
                            model_pred=model.predict(X_test)
                            r2_model=r2_score(y_test,model_pred)
                            #print(r2_model)
                            #print(X_test.head())                       
                            X_test_s=X_test.copy()
                            feat_shuff=X_test_s[feat].tolist()
                            np.random.shuffle(feat_shuff)
                            X_test_s.drop(feat,axis=1,inplace=True)
                            X_test_s.insert(ct,feat,feat_shuff)
                           # print(X_test_s.head())
                            ct+=1
                    
                            model_pred_shuffle=model.predict(X_test_s)               
                            model_shuffle_r2_acc=r2_score(y_test,model_pred_shuffle)   
                            #print(model_shuffle_r2_acc)
                            self.metricScore[feat].append((r2_model-model_shuffle_r2_acc)/r2_model*100)
                            #print(self.metricScore)
                            
               elif self.split==True:
                  
                   
                   for i in range(iterations):
                       X_train,X_test,y_train,y_test=train_test_split(self.predictors,self.target,
                                                                      test_size=self.testSize,
                                                                      random_state=np.random.randint(3000))
                       
                       ct=0
                       for feat in self.data.columns:                           
                                    
                            model.fit(X_train,y_train)
                            model_pred=model.predict(X_test)
                            r2_model=r2_score(y_test,model_pred)
                                                    
                            X_test_s=X_test.copy()
                            feat_shuff=X_test_s[feat].tolist()
                            np.random.shuffle(feat_shuff)
                            X_test_s.drop(feat,axis=1,inplace=True)
                            X_test_s.insert(ct,feat,feat_shuff)
                            #print(X_test_s.head())
                            ct+=1
                    
                            model_pred_shuffle=model.predict(X_test_s)               
                            model_shuffle_r2_acc=r2_score(y_test,model_pred_shuffle)                       
                            self.metricScore[feat].append((r2_model-model_shuffle_r2_acc)/r2_model*100)
                   
                   
           elif self.mse==True and self.r2_score==False:
               if self.split==False:
                   
                   for i in range(iterations):
                       X_train,X_test,y_train,y_test=train_test_split(self.data.drop(self.data.columns[-1],axis=1),
                                                                       self.data[self.data.columns[-1]],
                                                                       test_size=self.testSize, 
                                                                       random_state=np.random.randint(3000))
                       
                       ct=0
                       for feat in self.data.columns[:-1]:                           
                                    
                            model.fit(X_train,y_train)
                            model_pred=model.predict(X_test)
                            mse_model=mean_squared_error(y_test,model_pred)
                                                    
                            X_test_s=X_test.copy()
                            feat_shuff=X_test_s[feat].tolist()
                            np.random.shuffle(feat_shuff)
                            X_test_s.drop(feat,axis=1,inplace=True)
                            X_test_s.insert(ct,feat,feat_shuff)
                            #print(X_test_s.head())
                            ct+=1
                    
                            model_pred_shuffle=model.predict(X_test_s)               
                            model_shuffle_mse_acc=mean_squared_error(y_test,model_pred_shuffle)                       
                            self.metricScore[feat].append(((model_shuffle_mse_acc-mse_model)/mse_model)*100)
                            
               elif self.split==True:
                  
                   
                   for i in range(iterations):
                       X_train,X_test,y_train,y_test=train_test_split(self.predictors,self.target,
                                                                      test_size=self.testSize,
                                                                      random_state=np.random.randint(3000))
                       
                       ct=0
                       for feat in self.data.columns:                           
                                    
                            model.fit(X_train,y_train)
                            model_pred=model.predict(X_test)
                            mse_model=mean_squared_error(y_test,model_pred)
                                                    
                            X_test_s=X_test.copy()
                            feat_shuff=X_test_s[feat].tolist()
                            np.random.shuffle(feat_shuff)
                            X_test_s.drop(feat,axis=1,inplace=True)
                            X_test_s.insert(ct,feat,feat_shuff)
                            #print(X_test_s.head())
                            ct+=1
                    
                            model_pred_shuffle=model.predict(X_test_s)               
                            model_shuffle_mse_acc=mean_squared_error(y_test,model_pred_shuffle)                       
                            self.metricScore[feat].append(((model_shuffle_mse_acc-mse_model)/mse_model)*100)
           elif self.r2_score==True and self.mse==True:
               raise Exception('ERROR in metric selction. You have probably set both r2_score and mse to True.\n Set only one of the metrics to True.')
                   
               
       elif self.classModel==True and regModel==False:
           model=self.classifier
           
           if self.split==False:
                   
                   #print(X_train.head())
                   for i in range(iterations):
                       X_train,X_test,y_train,y_test=train_test_split(self.data.drop(self.data.columns[-1],axis=1),
                                                                       self.data[self.data.columns[-1]],
                                                                       test_size=self.testSize, 
                                                                      random_state=np.random.randint(3000))
                       model.fit(X_train,y_train)
                       model_pred=model.predict_proba(X_test)[:,1]
                       self.roc_auc['Original'].append(roc_auc_score(y_test,model_pred))
                       a,b,_=roc_curve(y_test,model_pred)
                       self.fpr['Original'].append(a)
                       self.tpr['Original'].append(b)
                       #print('ORIGINAL: {}'.format(self.fpr['Original']))
                       
                       ct=0
                       for feat in self.data.columns[:-1]:                           
                                    

                            X_test_s=X_test.copy()
                            feat_shuff=X_test_s[feat].tolist()
                            np.random.shuffle(feat_shuff)
                            X_test_s.drop(feat,axis=1,inplace=True)
                            X_test_s.insert(ct,feat,feat_shuff)
                          
                            ct+=1
                    
                            model_pred_shuffle=model.predict_proba(X_test_s)[:,1]
                            #print(model_pred_shuffle)
                            a,b,_=roc_curve(y_test,model_pred_shuffle)
                            #print('A=',a)
                            self.fpr[feat].append(a)
                            self.tpr[feat].append(b)
                            self.roc_auc[feat].append(roc_auc_score(y_test,model_pred_shuffle))
                            #print(roc_auc_score(y_test,model_pred_shuffle))
                            #print('{}={}'.format(feat,self.roc_auc[feat]))
                     
                         
           elif self.split==True:
                
               for i in range(iterations):
                   X_train,X_test,y_train,y_test=train_test_split(self.predictors,self.target,
                                                                      test_size=self.testSize,
                                                                      random_state=np.random.randint(3000))
                   model.fit(X_train,y_train)
                   model_pred=model.predict(X_test)
                   self.roc_auc['Original'].append(roc_auc_score(y_test,model_pred))
                   a,b,_==roc_curve(y_test,model_pred_shuffle)
                   self.fpr['Original'].append(a)
                   self.tpr['Original'].append(b)
                       
                   ct=0
                   for feat in self.data.columns:                           
                                
                        
                                                
                        X_test_s=X_test.copy()
                        feat_shuff=X_test_s[feat].tolist()
                        np.random.shuffle(feat_shuff)
                        X_test_s.drop(feat,axis=1,inplace=True)
                        X_test_s.insert(ct,feat,feat_shuff)
                        #print(X_test_s.head())
                        ct+=1
                
                        model_pred_shuffle=model.predict(X_test_s)               
                        self.fpr[feat],self.tpr[feat],_=roc_curve(y_test,model_pred_shuffle)
                        self.roc_auc[feat].append(roc_auc_score(y_test,model_pred_shuffle))
       else:
           raise Exception('ERROR in model selection. You have probably set both regModel and classModel to True.\nPossible solution: Set only one of the models to True')
       #print(self.roc_auc.keys())
       if regModel==True and boxPlot==False:
       
            return self.metricScore
        
       elif regModel==True and boxPlot==True:
            df_model=pd.DataFrame(self.metricScore)
            sns.set_style('whitegrid')
            plt.figure(figsize=(self.length,self.height))
            sns.boxplot(data=df_model)
            plt.xticks(fontsize=self.xticks_size)
            plt.yticks(fontsize=self.yticks_size)

            plt.title(title,fontsize=self.title_fontsize)
            
            if self.r2_score==True:
                plt.ylabel('Percentage decrease in {}'.format('R2 Score'),
                           fontsize=self.y_fontsize)
                plt.xlabel("Features",fontsize=self.x_fontsize)
                
            elif self.mse==True:
                plt.ylabel("Percenate increase in {}".format('MSE'),
                           fontsize=self.y_fontsize)
                plt.xlabel('Features',fontsize=self.x_fontsize)
             
            plt.show()
            
       elif classModel==True and roc==True:
           n=len(self.roc_auc.keys())
           colors = iter(cm.gist_ncar(np.linspace(0, 1, n)))
           plt.figure(figsize=(self.length,self.height))
           for feat in list(self.roc_auc.keys())[1:]:
               
               plt.plot(self.fpr[feat][0], self.tpr[feat][0], 
                        label='Shuffling  {} (Area={:.3f})'.format(feat.upper(),self.roc_auc[feat][0]),
                        linestyle='-',lw=2,color=next(colors))
           
           plt.plot(np.linspace(0,1,10),np.linspace(0,1,10),linestyle='--',lw=3, label='No learning')
               #print(self.roc_auc[feat])
           plt.plot(self.fpr['Original'][0],self.tpr['Original'][0],
                    label='ROC curve for the ORIGINAL model (Area={:.3f})'.format(self.roc_auc['Original'][0]),
                    linestyle=':',lw=4,color='r')
           plt.legend(loc='lower right')
           plt.title(title,fontsize=self.title_fontsize)
           plt.xlabel('False Positive Rate',fontsize=self.x_fontsize)
           plt.ylabel('True Positive Rate',fontsize=self.y_fontsize)
       #print(self.fpr['Original'])
    
               
       
       
       
               
               
               
      
           
          
                   
                    
                            
                    
            
            
                    
 #Testing the Code
 
from sklearn.datasets import load_breast_cancer
 
cancer=load_breast_cancer()

#print(cancer.data)
#print(cancer.target)

df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
cols=df.columns[::3]
df=df[cols]
#df.drop(cols,axis=1,inplace=True)
df['target']=cancer.target

#print(df.head())
 
from sklearn.linear_model import LogisticRegression

df2=pd.DataFrame(np.random.randn(20,5),columns=['a','b','c','d','e'])
df2['target']=[0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1]
#print(df2)

feat_sel=FeatureSelection(df, classifier=LogisticRegression(penalty='l2'))

out=feat_sel.feature_selection(classModel=True,regModel=False,
                               roc=True,
                               boxPlot=False,
                               split=False,
                               iterations=1,
                               length=14,
                               height=8,
                               title='ROC Curves After Shuffling',
                               title_fontsize=22,
                               x_fontsize=16,
                               y_fontsize=16)

out
                        
                        
                    
                 
            














ree=pd.read_excel('ree_body_data.xls',sheet_name='Edited Data')
females=ree[ree['Sex (1M,2F)']==0]
females3=females[['Height (m)','AT','SM','Brain mass', 'liver mass','heart mass','kidney masses','REE']]
females3.columns=['C1','C2','C3','C4','C5','C6','C7','target']
#print(females3.head())
females3.drop(298,inplace=True)
#print(females3.shape)
outlier=Outliers(females3)

females3,outliers=outlier.remove_outliers()

#print(outliers)
#
#print(females3.shape)

feat_sel=FeatureSelection(females3,r2_score=True,
                       mse=False,
                       regressor=LinearRegression())

r2=feat_sel.feature_selection(regModel=True,
                              classModel=False,
                              boxPlot=True,
                              length=12,
                              x_fontsize=18,
                              y_fontsize=16,
                              xticks_size=14,
                              yticks_size=14,
                              title='Feature Selection For XXX',
                              title_fontsize=18,
                              iterations=50)


r2









