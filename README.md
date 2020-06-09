## LOGISTIC REGRESSION:
- - - - 
( I’m using Python 3.7 and Jupyter Notebook. To run code without installation of Jupyter it is possible to run in online here:
—> https://jupyter.org/try
1. Choose JupyterLab
2. Choose python 3
3. Load this file just by dropping it in place where are other sample files )
- - - - 

Regression analysis in general are methods used for prediction.

- Linear regression shows a linear approximation of a relationship between two (or more) variables.

- Logistic regression is useful when we want to predict the probability of certain event with data
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

#### In this case I will provide MLE.
- - - - 
Background:

Dataset I made contains scores from exam of 100 students. Students could get from 0 to 200
points. The dataset includes only those student who passed the exam, i.e. those who obtained
above 100 points because only them can apply for higher school.
Yes and No stands for if they were accepted by school or no.
Goal here is to predict if the student will get to school based on the results of the exam.
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
```

2. Loading and viewing data:
```
raw_data = pd.read_csv(‘dataset_school.csv’)
raw_data
```
3. Changing ‘no’ entries into zeros and ‘yes’ into ones:
```
data = raw_data.copy()
data['School'] = data['School'].map({'Yes':1, ‘No':0})

data
```

4. Declare the dependent & independent variables:
```
y = data[‘School']
x1 = data['Score']
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
Current function value: 0.
Iterations 9
```

This means that the regression is fitted.
Currect function value shows the value of objective function at the 9th iteration. It refers to this
that SM (StatModels) uses the machine learning algorithm to fit the regression.
In practice we used it to check if after some number of iterations model is still learning - when we
using SM maximum number of iterations is 35.

4. Summarizing:

```
results_log.summary()
```

SM provides really good summaries.

**Method:** As a method we used **MLE** - Maximum likelihood estimation. It is based on Likelihood
function.
- Likeihood function estimates how likely is that the model shows real relationship between
    variables so the higher value of likelihood function the higher probability that model is correct
So, MLE is maximizing this function. In this case computer looking for model for which likelihood
is the highest.

**Log-likelohood** : Usually this value is negative (not always) and in practice the bigger it is is the
better.


**L-L-Null** : It is log likelihood-null. This is value that we get by the constant that we add earlier. We
using this value to see if model has explanatory power by comparing the log-likehood to this
value.

**LLR p-value** : (log likelihood ratio): measures significance of model. When this value is very low
the model is significant.

**R-squared** : Unlike than previously r-squared is not defined so easily in this case. This
one from our summary is called McFadden’s R-squared and it should be around between 0.2 -
0.4. We using this for comparing variations in the same model.

The next table is coefficient table:
Mathematically logit model for our data is:

![equation](http://www.sciweavers.org/upload/Tex2Img_1591743116/render.png)

![equation](http://www.sciweavers.org/upload/Tex2Img_1591743185/render.png)—> probability of event occurring

![equation](http://www.sciweavers.org/upload/Tex2Img_1591743221/render.png)—> probability of event NOT occurring

5. We can visualize result on a scatter plot.

```
plt.scatter(x1,y,color = 'C0')
plt.xlabel('Score', fontsize = 20)
plt.ylabel('School', fontsize = 20)
plt.show()
```

6. Additionally I’m trying with linear regression with method OLS:

```
x = sm.add_constant(x1)
reg_lin=sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y, color='blue')
y_hat = x1*results_lin.params[1]+results_lin.params[0]
plt.plot(x1,y_hat,lw=2,color="red")
plt.xlabel("Score")
plt.ylabel("School")
plt.show()
```

Regression don’t fit very well in this case.

