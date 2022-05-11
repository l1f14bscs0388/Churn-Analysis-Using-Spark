#Loading the Data and Pre-Processing

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
pd.options.display.mpl_style = 'default'
import numpy as np
from pyspark.sql.types import *
from pyspark.sql import Row
import seaborn as sns
from pyspark import SparkContext
from pyspark import SQLContext

sc = SparkContext()
spark = SQLContext(sc)

#Import the Data

ch = spark.read.csv("C:/churn.csv",header=True,inferSchema=True)

ch

#Data Preprocessing

ch['Churn']= ch['Churn'].astype('category')
ch['Intl Plan']= ch['Intl Plan'].astype('category')
ch['VMail Plan']= ch['VMail Plan'].astype('category')

#Exploratory Data Analysis

ch.describe()

#Histogram for Day minutes spent by customers
plt.hist(ch['Day Mins'], bins= 10, facecolor= 'tan')
plt.xlabel('Total Day Minutes')
plt.ylabel('No. of Customers')
plt.show()

import seaborn as sns
import seaborn as sns
g = sns.FacetGrid(ch, col="Churn")
g.map(plt.hist, "Day Mins")

#Number of customers opt voice mail plan
ch['VMail Plan'].value_counts()

sns.set(style="whitegrid", color_codes=True)
sns.countplot(x="VMail Plan", hue= "Churn", data=ch)


#International Plan opt by customer

ch['Intl Plan'].value_counts()
sns.countplot(x="Intl Plan", hue= "Churn", data=ch)

#Areawise churner and non-churner

ch['Area Code']= ch['Area Code'].astype('category')
sns.countplot(x="Area Code", hue= "Churn", data=ch)

#Correlation Matrix
ch.corr('pearson')
#Correlation between Predicting Variable and independent variable.
ch.corr()["Churn"]

