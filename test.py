#!/usr/bin/env python
# coding: utf-8

# # Spectrum Lite Risk Model

# In[1]:

#Import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import pickle
import glob
import os


# # Import the new dataset for prediction

df1=pd.read_csv("Daily+New+CP+Report (13)_full_2022-11-29_22-14-57.csv")

column_list=['Record Id','Account Name','Annual Revenue','Revenue (US Dollars)','D-U-N-S Number','Year Started','Last Update Date','Employee Count Total','Major Industry Category','Major Industry Category Name','BEMFAB (Marketability)','Line of Business']

df2=df1[column_list]

#Data Preprocessing to feed the model

df2['Revenue (US Dollars)']=np.where(df2['Revenue (US Dollars)']==np.nan,df2['Annual Revenue'],df2['Revenue (US Dollars)'])
df2['Revenue (US Dollars)']=np.where(df2['Revenue (US Dollars)']==0,df2['Annual Revenue'],df2['Revenue (US Dollars)'])
df2['Revenue (US Dollars)']=df2['Revenue (US Dollars)'].fillna(0)

df2.dropna().drop_duplicates()

final_df=df2.drop_duplicates()
final_df

final_df['Year Started']=final_df['Year Started'].fillna(final_df['Year Started'].mode()[0])
final_df['Last Update Date']=final_df['Last Update Date'].fillna(final_df['Last Update Date'].mode()[0])
final_df['Employee Count Total']=final_df['Employee Count Total'].fillna(final_df['Employee Count Total'].mean())


# # Null Hypothesis for a good Customer Partner(H0)

# #1.The Annual Revenue must be high
# 
# #2.The cp must be old in terms of age
# 
# #3.The must updated recently
# 
# #4.No of employee must be at higher side


final_df['Year Started']=final_df['Year Started'].astype('int')
final_df['Last Update Date']=final_df['Last Update Date'].astype('int')


# s = "20120213"
# # you could also import date instead of datetime and use that.
# date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))

new_date=[]

for date in final_df['Last Update Date']:
    s = str(date)
    s_datetime =datetime.strptime(s, '%Y%m%d')
    new_date.append(s_datetime)


final_df['Last Update Date']=new_date

#final_df['BEMFAB (Marketability)'].value_counts()

# Add a day_since column showing the difference between last purchase and a basedate
basedate = datetime.now()
final_df['Recency'] = (basedate - final_df['Last Update Date']).dt.days
final_df['Year Started']=np.where(((final_df['Year Started']<1800) | (final_df['Year Started']>2022)),2022,final_df['Year Started']  )


# Find the age of the company


old_comp=[]
for dt in final_df['Year Started']:
#     print(dt)
    s = str(dt)
    date = datetime(year=int(s[0:4]), month=int('01'), day=int('01'))
#     old_comp.append(date)
    basedate = datetime.now()
    age = (basedate - date).days
    old_comp.append(age)


final_df['Age']=old_comp

final_df['BEMFAB (Marketability)']=np.where((final_df['BEMFAB (Marketability)']=='Marketable'),1,0)

#final_df['BEMFAB (Marketability)'].value_counts()


# # Model Prediction

##Load the saved pickle file generated from training.py file

with open('model.pkl','rb') as load_file:
    model=pickle.load(load_file)

with open('quantiles.pkl','rb') as load_file_1:
    quantiles=pickle.load(load_file_1)

print("<<<Model loading Sucessful>>>")


########Assigning weatage 


# We create five classes for the RFM segmentation since, being high revenue is good, high Employee Count Total is good ,High recency is bad and high age is good for bussiness entity.

model_df=final_df[['Account Name','Revenue (US Dollars)','Employee Count Total','Recency','Age','BEMFAB (Marketability)']]

# # Creating function to give weightage accordance with +ve and -ve corr
# 
# We create five classes for the RFM segmentation since, being high revenue is good, high Employee Count Total is good ,High recency is bad and high age is good for bussiness entity.

# For positive correlation
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
# For negetive correlation
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

#For positive corerlation
model_df['Rev_Quartile'] = model_df['Revenue (US Dollars)'].apply(RClass, args=('Revenue (US Dollars)',quantiles,))
model_df['Emp_Quartile'] = model_df['Employee Count Total'].apply(RClass, args=('Employee Count Total',quantiles,))
model_df['Age_Quartile'] = model_df['Age'].apply(RClass, args=('Age',quantiles,))

#For negetive corerlation
model_df['Rncy_Quartile'] = model_df['Recency'].apply(FMClass, args=('Recency',quantiles,))

model_df['Confidence Indicator']=(model_df['BEMFAB (Marketability)']+model_df['Rev_Quartile']+
                            model_df['Emp_Quartile']+model_df['Age_Quartile']+model_df['Rncy_Quartile'])/17

# # Method-2 (K-Mean Clustering)

# Accoring to the elbow method we are safe to say that there are 3 cluster forming.


X_test = model_df[['BEMFAB (Marketability)', 'Rev_Quartile', 'Emp_Quartile','Age_Quartile', 'Rncy_Quartile']]

y_predict= model.predict(X_test)

model_df['Cluster']=y_predict
model_df['Safe Class']=1


# ###From the above value we have the idea that 
# 
# 'Highly Qualified' ==> (Class 2)
# 
# 'Avarage Qualified' ==> (Class 1)
# 
# 'Poorly Qualified' ==> (Class 0)

model_df['Safe Class']=np.where(model_df['Cluster']==2,'Highly Qualified',model_df['Safe Class'])
model_df['Safe Class']=np.where(model_df['Cluster']==1,'Average Qualified',model_df['Safe Class'])
model_df['Safe Class']=np.where(model_df['Cluster']==0,'Poorly Qualified',model_df['Safe Class'])


from datetime import datetime

today_date=datetime.now().date()

model_df['Date']=today_date

result_df=model_df.drop('Cluster',axis=1)

#from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
print(f"filename_{date}")


# # Daily result to be append to the main result file

file_name = 'Spectrum_scorecard_'+date+'.xlsx'

saved_data=pd.read_excel('Testing_data.xlsx',)

saved_data=saved_data.drop('Unnamed: 0',axis=1)

saved_data=saved_data.append(result_df,ignore_index=True)

saved_data.to_excel('Testing_data.xlsx')


##Find the factors which affecting the cluster badly(Poorly classified and average qualified)
saved_data.iloc[:,6:10]

saved_data['Lowest'] = saved_data.iloc[:,6:10].idxmin(axis=1)

saved_data['Remark'] = np.nan
saved_data.loc[(((saved_data['Safe Class']=='Poorly Qualified') | (saved_data['Safe Class']=='Average Qualified')) & (saved_data['Lowest']=='Rev_Quartile')), 'Remark'] = 'Low Revenue'
saved_data.loc[(((saved_data['Safe Class']=='Poorly Qualified') | (saved_data['Safe Class']=='Average Qualified')) & (saved_data['Lowest']=='Emp_Quartile')), 'Remark'] = 'Low Employee Count'
saved_data.loc[(((saved_data['Safe Class']=='Poorly Qualified') | (saved_data['Safe Class']=='Average Qualified')) & (saved_data['Lowest']=='Age_Quartile')), 'Remark'] = 'Low Age Company'
saved_data.loc[(((saved_data['Safe Class']=='Poorly Qualified') | (saved_data['Safe Class']=='Average Qualified')) & (saved_data['Lowest']=='Rncy_Quartile')), 'Remark'] = 'Low Recency'

saved_data=saved_data.drop('Lowest',axis=1)

# saved_data.drop('Unnamed: 0',axis=1)

# saved_data=saved_data.drop_duplicates()

# # Model Prediction

date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
print(f"filename_{date}")

file_name = 'Spectrum_scorecard_'+date+'.xlsx'

saved_data.to_excel(file_name)

print("Final Result Generated>>>>>")


