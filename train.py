#!/usr/bin/env python
# coding: utf-8

# # Spectrum Lite Risk Model

#Import all the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import pickle
import warnings
warnings.filterwarnings('ignore')

##Import the training dataset###

df2=pd.read_excel('Past+Due+Verify+List_full_2022-10-04_20-17-43_.xlsx')

#df2.columns

column_list=['Record Id','Account Name','Annual Revenue','D-U-N-S Number','Year Started','Last Update Date','Employee Count Total','Revenue (US Dollars)','Major Industry Category','Major Industry Category Name','BEMFAB (Marketability)','Line of Business']

df2=df2[column_list]

#Exploratory Data Analysis

df2['Revenue (US Dollars)']=np.where(df2['Revenue (US Dollars)']==np.nan,df2['Annual Revenue'],df2['Revenue (US Dollars)'])
df2['Revenue (US Dollars)']=np.where(df2['Revenue (US Dollars)']==0,df2['Annual Revenue'],df2['Revenue (US Dollars)'])
df2['Revenue (US Dollars)']=df2['Revenue (US Dollars)'].fillna(0)
df2.isna().sum()


#Missing value handling

df2.isna().sum()


df2.dropna().drop_duplicates()

final_df=df2.drop_duplicates()


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


# Add a day_since column showing the difference between last purchase and a basedate
basedate = datetime.now()
final_df['Recency'] = (basedate - final_df['Last Update Date']).dt.days
final_df['Year Started']=np.where(((final_df['Year Started']<1800) | (final_df['Year Started']>2022)),2022,final_df['Year Started']  )

old_comp=[]
for dt in final_df['Year Started']:
#     print(dt)
    s = str(dt)
    date = datetime(year=int(s[0:4]), month=int('01'), day=int('01'))
#     old_comp.append(date)
    basedate = datetime.now()
    age = (basedate - date).days
    old_comp.append(age)
    

###To find the age of the company
final_df['Age']=old_comp


#Create marketable column 


final_df['BEMFAB (Marketability)']=np.where((final_df['BEMFAB (Marketability)']=='Marketable'),1,0)


final_df['BEMFAB (Marketability)'].value_counts()

###Create quntiles for the model to train


quantiles = final_df.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
quantiles


########Assigning weatage 


# We create five classes for the RFM segmentation since, being high revenue is good, high Employee Count Total is good ,High recency is bad and high age is good for bussiness entity.

model_df=final_df[['Account Name','Revenue (US Dollars)','Employee Count Total','Recency','Age','BEMFAB (Marketability)']]

model_df['Revenue (US Dollars)']=model_df['Revenue (US Dollars)'].astype('int')


# # #Creating function to give weightage to
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



# Split into training and testing sets 
X = model_df[['BEMFAB (Marketability)', 'Rev_Quartile', 'Emp_Quartile','Age_Quartile', 'Rncy_Quartile']]


# In[239]:


#finding optimal number of clusters using the elbow method  
  
#wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
#for i in range(1, 11):  
  #  kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
 #   kmeans.fit(X)  
#    wcss_list.append(kmeans.inertia_)  
#plt.plot(range(1, 11), wcss_list)  
#plt.title('The Elobw Method Graph')  
#plt.xlabel('Number of clusters(k)')  
#plt.ylabel('wcss_list')  
#plt.show()  


# Accoring to the elbow method we are safe to say that there are 3 cluster forming.

# In[240]:


#training the K-means model on a dataset  
kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
y_predict= kmeans.fit_predict(X)


kmeans_1 = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
kmeans_1.fit(X)

model_df['Cluster']=y_predict

cluster_labels = kmeans.labels_
#cluster_labels

# Scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
df_rfm_normal = scaler.transform(X)

df_rfm_normal = pd.DataFrame(X, index=X.index, columns=X.columns)

# Check result after standardising
df_rfm_normal.describe().round(3)


# In[245]:


df_rfm_k3 = X.assign(Cluster = cluster_labels)

def relative_importance(df_rfm_kmeans, df_rfm_original):
    '''
    Calculate relative importance of segment attributes and plot heatmap
    '''
    # Calculate average RFM values for each cluster
    cluster_avg = df_rfm_kmeans.groupby(['Cluster']).mean() 

    # Calculate average RFM values for the total customer population
    population_avg = X.mean()

    # Calculate relative importance of cluster's attribute value compared to population
    relative_imp = cluster_avg / population_avg - 1

    #sns.heatmap(data=relative_imp, annot=True, fmt='.2f')
    
    return relative_imp

#plt.figure(figsize=(9, 9))

#plt.subplot(3, 1, 1)
#plt.title('Relative Importance of K-Means = 3')
imp_score=relative_importance(df_rfm_k3, X)

model_df=model_df.assign(Cluster = cluster_labels)

score_dict=imp_score.sum(axis=1).to_dict()

score_list=sorted(score_dict, key = score_dict.get,reverse=True)

model_df['Safe Class']=1


# ###From the above value we have the idea that 
# 
# (Safe) ==> (Class 1)
# 
# (Avarage Safe) ==> (Class 2)
# 
# (UnSafe) ==> (Class 3)

# In[257]:


model_df['Safe Class']=np.where(model_df['Cluster']==score_list[0],'Highly Qualified',model_df['Safe Class'])
model_df['Safe Class']=np.where(model_df['Cluster']==score_list[1],'Avarage Qualified',model_df['Safe Class'])
model_df['Safe Class']=np.where(model_df['Cluster']==score_list[2],'Poorly Qualified',model_df['Safe Class'])

model_df['Safe Class'].value_counts()

###Pickling the model

with open('model.pkl','wb') as dump_file:
    pickle.dump(kmeans_1,dump_file)

with open('quantiles.pkl','wb') as dump_file_1:
    pickle.dump(quantiles,dump_file_1)


print("##########Model Training Complete########")


