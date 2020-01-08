# DataAnalysisToolbox

The methods in this Python Module can be used to perform the following tasks:

# Data Cleaning:
Specifically:

  (1) <b><font size='6'>Drop and Impute</font></b>
  
  This method will drop subjects and features with 'too many' missing values and impute the rest 
  
  <b>How to use:</b>
  
 ```python
    from minitoolboxVB import DropImpute
     
            drop=DropImpute(dataframe)
            drop.drop_subjects(3,inplace=True)
            drop.impute_values(5,inplace=True)
            
 ```
            
   Applying these methods in this order, will first drop subjects who are missing three or more values; it will drop features that miss 5% of values or more, and it will impute the rest by first building an empirical distribution for each feature and then imputing the missing values by values randomly sampled from the corresponding empirical distribution. 
  
  
  (2) <b><font size='6'>Remove Outliers</font></b>
  
   This method will remove all the values that lie below Q1-1.5IQR and above Q3+1.5IQR
  
 <b>How to use:</b>
 
 ```python
      
      from minitoolboxVB import Outliers
      
          outliers=Outliers(dataframe)
          outliers.remove_outliers()
 ```
    
    
     
  
# Feature Selection

The methods will perfor feature selection for both regression and classification models. 
First, it will randomly split the data in a training and test set. Then, it will train the model on the training set. Next, it will shuffle the values of each feature in the test set and measure one of the prespecified metrics to observe the decrease in the predictive performance of the model.

 <b>How to use:</b>
 
 ```python
    
    from minitoolboxVB import FeatureSelection
    
      feat_sel=FeatureSelection(df, classifier=LogisticRegression(solver='liblinear',penalty='l2'))

      out=feat_sel.feature_selection(classModel=True,regModel=False,
                                                           roc=True,
                                                           boxPlot=False,
                                                           split=False,
                                                           iterations=1,
                                                           length=14,
                                                           height=8,
                                                           title='Feature Importance: ROC Curves After Shuffling',
                                                           title_fontsize=22,
                                                           x_fontsize=16,
                                                           y_fontsize=16)
```
    
    
