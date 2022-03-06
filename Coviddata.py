import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as pt


covdata = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-01-2021.csv")
covdata.head(20)

covdata.info()

covdata.columns

covdata1 = covdata[[ 'Lat', 'Long_', 'Confirmed', 'Deaths', 'Recovered', 'Active',
        'Incident_Rate', 'Case_Fatality_Ratio']]
covdata1.head(10)
tarvar= covdata1["Case_Fatality_Ratio"] 
feamat = covdata1.iloc[:,0:7]
covdata1.isna().sum
covdata1 = covdata1.fillna(method = "ffill")
covdata1.head(5)
covdata1.isna().sum()

covdata1.dtypes

covdata1.info()

correlation = covdata1.corr()
correlation


pt.figure(figsize = (13,15))
sb.heatmap(correlation, annot = True)
pt.show()

#absolute corelation on the target variable
corr_target = abs(covdata1["Case_Fatality_Ratio"])
corr_target
feamat.isna().sum()
feamat = feamat.fillna(method = "ffill")


import statsmodels.api as sm
c1 = sm.add_constant(x)
model = sm.OLS(feamat, c1).fit()


