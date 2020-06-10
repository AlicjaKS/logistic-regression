## USING LOGISTIC REGRESSION IN ML TO PREDICT VALUES:
- - - - 
( I’m using Python 3.7 and Jupyter Notebook. To run code without installation of Jupyter it is possible to run in online here:
—> https://jupyter.org/try
1. Choose JupyterLab
2. Choose python 3
3. Load this file just by dropping it in place where are other sample files )
- - - - 

Regression analysis in general are methods used for prediction.

1. Linear regression shows a linear approximation of a relationship between two (or more) variables.

2. Logistic regression is useful when we want to predict the probability of certain event with data
that can give possible categorical outcome (not numerical), for example:
 - yes / no
 - will buy / won’t buy
 - 1 / 0

**We have several method for finding the regression line:** 

1. OLS - ordinary least squares
2. GLS - generalized least squares
3. MLE - maximum likelihood estimation
4. Bayesian regression
5. Kernel regression
6. Gaussian progress regression

#### In this case I will provide MLE (and just to check OLS).
- - - - 
Background:

Dataset I made contains scores from exam of 99 students. Students could get from 0 to 200
points. The dataset includes only those student who passed the exam, i.e. those who obtained
above 100 points because only them can apply for higher school.
Yes and No stands for if they were accepted by school or no. We also have information about their gender. 
Goal here is to predict if the student will get to school based on the results of the exam and their gender.
- - - - 
### Step by step:

1. Importing relevant packages:
```
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats
```

```
stats.chisqprob = lambda chisq, df:stats.chi2.sf (chisq, df)
```

2. Loading and viewing data:
```
raw_data = pd.read_csv(‘dataset_school.csv’)
raw_data
```
3. Changing ‘no’ entries into zeros and ‘yes’ into ones and also gender F = 1, M = 0:
```
data = raw_data.copy()
data['School'] = data['School'].map({'Yes':1, ‘No':0})
data['Gender'] = data['Gender'].map({'F':1, 'M':0})

data
```

4.1. Declare the dependent & independent variables - first, checking if which **gender** is more likely to get to the school:
```
y = data[‘School']
x1 = data['Gender']
```

### LOGISTIC REGRESSION:

1. Adding constant:
```
x = sm.add_constant(x1)
```
2. Declaring regression variable, defining calculations and fitting regression:
```
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
```

We using method = Logit

3. You should get output:

```
Optimization terminated successfully.
Current function value: 0.683133
Iterations 4
```

This means that the regression is fitted.
Currect function value shows the value of objective function at the 9th iteration. It refers to this that SM (StatModels) uses the machine learning algorithm to fit the regression.
In practice we used it to check if after some number of iterations model is still learning - when we using SM maximum number of iterations is 35.

4. Summarizing:

```
results_log.summary()
```

SM provides really good summaries.

**Method:** As a method we used **MLE** - Maximum likelihood estimation. It is based on Likelihoodfunction.
- Likeihood function estimates how likely is that the model shows real relationship between variables so the higher value of likelihood function the higher probability that model is correct
So, MLE is maximizing this function. In this case computer looking for model for which likelihood is the highest.

**Log-likelohood** : Usually this value is negative (not always) and in practice the bigger it is is the better.

**L-L-Null** : It is log likelihood-null. This is value that we get by the constant that we add earlier. We using this value to see if model has explanatory power by comparing the log-likehood to this value.

**LLR p-value** : (log likelihood ratio): measures significance of model. When this value is very low the model is significant. *In this case it is not low (0.6028) so it's showing that gender have no influence to being admitted to school.* The same result we see in coef table in P>|z| column. 

**R-squared** : Unlike than previously r-squared is not defined so easily in this case. This one from our summary is called McFadden’s R-squared and it should be around between 0.2 - 0.4. We using this for comparing variations in the same model.

4.2. Then let's check only influence of the **score** just by changing 
x1 variable to 'Score' and rerun the code. 

```
y = data['School']
x1 = data['Score']
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log_SCORE = reg_log.fit()
```
Now we get output: 
```
Optimization terminated successfully.
         Current function value: 0.174638
         Iterations 9
```    
We watch at the summary in the same way. Now model is significant. 

The next table is coefficient table:
Mathematically logit model for our data is:

![equation](http://www.sciweavers.org/upload/Tex2Img_1591743116/render.png)

![equation](http://www.sciweavers.org/upload/Tex2Img_1591743185/render.png)—> probability of event occurring

![equation](http://www.sciweavers.org/upload/Tex2Img_1591743221/render.png)—> probability of event NOT occurring

5. We can also visualize result on a scatter plot.

```
plt.scatter(x1,y,color = 'C0')
plt.xlabel('Score', fontsize = 20)
plt.ylabel('School', fontsize = 20)
plt.show()
```

4.3. Next, we can try with both variables - gender and score in the same way: 
```
y = data['School']
x1 = data[['Score', 'Gender']]
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log_BOTH = reg_log.fit()
results_log_BOTH.summary()
```

Log-likelihood doesnt change much from this when we used only Score but is much better when we were using only Gender as predictor. Variable gender is still not significant. 

! Since gender variable is not significant we shoudnt analyze in this way but I want only show interpretation: 
```
np.exp(-0.6103)
```
Output = 0.5431

So when we have two students with the same score from exam, females have 0.5 times lower chanses to go to school. 

In next steps I will not include gender variable. 

6. Calculating accuracy of the model: 
```
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
results_log_SCORE.predict()
```

In output we have values like 0.00, 1.00 but also 0.89. This values are "pi" in our model so it is probability of being admitted by school. 
Values below 0.5 we interpret that there is less than 50% chanses of being admitted so I will round them to 0. The same way with values above 0.5:


Now take a look on actual values: 
```
np.array(data['School'])
```
Now using StatsModels we can make a table that compare predicted values and actual values and using Pandas we can format it to look better :) : 
```
results_log_SCORE.pred_table()

matrix = pd.DataFrame(results_log_SCORE.pred_table())
matrix.columns = ['Pred 0', 'Pred 1']
matrix = matrix.rename(index={0:'Actual 0', 1:'Actual 1'})
matrix
```
As an output we have small matrix with 4 values called **CONFUSION MATRIX**. 
How to interpret it? 

Value Pred 0/Actual 0 = 37.0 --> it is value where model predict 0 and in fact it is 0 
And for 53 observations model predict value as 1 and it have right. 
Those are situations where our model was righ so we can say that: model made an accurate prediction in 90 from 99 cases. 90/99 = 0,909 = **99.1% accuracy** 
Here is code for calculating it: 

```
confusion = np.array(matrix)
accuracy_train = (confusion[0,0]+confusion[1,1])/confusion.sum()
accuracy_train
```

7. Testing the model
Testing is done on a dataset the model has never seen so I split original dataset and save 20 observations in another file:
```
test = pd.read_csv('TESTdataset_school.csv')
test
```
We have to change caterogical variables into numerical as before: 
```
test['School'] = test['School'].map({'Yes':1, 'No':0})
test['Gender'] = test ['Gender'].map({'F':1, 'M':0})
test
```
Now we split this table into two: 
One that have actual outcome observed
And second with with only gender and score from exam: 
```
test_actual = test['School']
test_data = test.drop(['School'], axis = 1)
test_data = sm.add_constant(test_data)
test_data
```
We have to look at confusion matrix but this time we have to do it manually: 

```
def confusion_matrix(data, actual_values, model):
   
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy
cm = confusion_matrix(test_data, test_actual, results_log_SCORE)
cm
```
Results are confusion matrix and accuracy. The opposite of accuracy is **misslacassification rate** 
In our case it is 10%:
```
print( 'Missclasification rate: ' +str ((1+1)/20))
```


