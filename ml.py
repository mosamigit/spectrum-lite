
# # Spectrum Lite Risk Model

# Import all the libraries
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import MinMaxScaler
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

confClass_1 = float(os.getenv("CONFIDENCE_THRESH_1"))
confClass_2 = float(os.getenv("CONFIDENCE_THRESH_2"))
confClass_3 = float(os.getenv("CONFIDENCE_THRESH_3"))
confClass_4 = float(os.getenv("CONFIDENCE_THRESH_4"))


# # Import the new dataset for prediction

def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        print(f"clean_currency - REPLACING x value: {x}")
        return (x.replace('$', '').replace(',', ''))
    return (x)


def update_safe_class_1(row):

    if row["Confidence Indicator"] > confClass_1 and row["riskLevel"] == "HIGH":
        return "MEDIUM"
    else:
        return row["riskLevel"]


def update_safe_class_2(row):

    if row["Confidence Indicator"] < confClass_2 and row["riskLevel"] == "LOW":
        return "MEDIUM"
    else:
        return row["riskLevel"]


def update_safe_class_3(row):

    if row["Confidence Indicator"] > confClass_3 and row["riskLevel"] == "MEDIUM":
        return "LOW"
    elif row["Confidence Indicator"] < confClass_4 and row["riskLevel"] == "MEDIUM":
        return "HIGH"
    else:
        return row["riskLevel"]
    

def calculate_score(dataframe):
    # Define the weights for each criterion
    dataframe[['Revenue (US Dollars)','Employee Count Total',
               'Recency','Age']]=dataframe[['Revenue (US Dollars)','Employee Count Total'
                                            ,'Recency','Age']].astype('float')
    
    dataframe['Revenue (US Dollars)'] = pd.to_numeric(dataframe['Revenue (US Dollars)'], errors='coerce').fillna(0)
    dataframe['Employee Count Total'] = pd.to_numeric(dataframe['Employee Count Total'], errors='coerce').fillna(0)
    dataframe['Age'] = pd.to_numeric(dataframe['Age'], errors='coerce').fillna(0)
    dataframe['Recency'] = pd.to_numeric(dataframe['Recency'], errors='coerce').fillna(0)

    # Calculate weightage for Revenue based on revenue range
    revenue_ranges = [0,500_000 ,1_000_000, 5_000_000, 10_000_000, 25_000_000,
                       50_000_000, 100_000_000, 250_000_000, 500_000_000, 1_000_000_000, float('inf')]
    revenue_weightages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dataframe['Revenue Weightage'] = pd.cut(dataframe['Revenue (US Dollars)'],
                                             bins=revenue_ranges, labels=revenue_weightages)

    # Calculate weightage for Employee Count based on employee count range
    emp_count_ranges = [0, 25, 50, 100, 500, 1000, 5000, 25000, 100000, float('inf')]
    emp_count_weightages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataframe['Employee Count Weightage'] = pd.cut(dataframe['Employee Count Total'],
                                                    bins=emp_count_ranges, labels=emp_count_weightages)

    # Calculate weightage for Recency based on number of days since last activity
    recency_ranges = [0,180, 365, 730, 1095, 1825, 3650, 7300, 10950, 18250, float('inf')]
    recency_weightages = [10,9, 8, 7, 6, 5, 4, 3, 2, 1]
    dataframe['Recency Weightage'] = pd.cut(dataframe['Recency'],
                                             bins=recency_ranges, labels=recency_weightages)

    # Calculate weightage for Age based on number of days since account creation
    age_ranges = [0, 365, 730, 1095, 1825, 3650, 7300, 10950, 18250, float('inf')]
    age_weightages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataframe['Age Weightage'] = pd.cut(dataframe['Age'],
                                         bins=age_ranges, labels=age_weightages)
    
    
    dataframe[['Revenue Weightage','Employee Count Weightage',
               'Recency Weightage','Age Weightage']]=dataframe[['Revenue Weightage','Employee Count Weightage',
                                                                'Recency Weightage','Age Weightage']].astype('float')

    # Define the weights for each criterion
    weights = {'Revenue':0.4,'Employee': 0.25, 'Recency': 0.3, 'Age': 0.2, 'Marketability': 0.2}

    # Calculate the score for each data point
    dataframe['Score'] = (weights['Revenue']*dataframe['Revenue Weightage']+\
                  dataframe['Employee Count Weightage']) * weights['Employee'] + \
                  dataframe['Recency Weightage'] * weights['Recency'] + \
                  dataframe['Age Weightage'] * weights['Age'] + \
                  dataframe['BEMFAB (Marketability)'] * weights['Marketability']
    
    dataframe['Score']=np.where(dataframe['riskLevel']=='LOW',(dataframe['Score']*1.3),dataframe['Score'])
    dataframe['Score']=np.where(dataframe['riskLevel']=='MEDIUM',(dataframe['Score']*0.7),dataframe['Score'])
    dataframe['Score']=np.where(dataframe['riskLevel']=='HIGH',(dataframe['Score']*0.5),dataframe['Score'])

    
    # Normalize the score using MinMaxScaler

    with open('scaler.pkl','rb') as load_file:
        scaler_model=pickle.load(load_file)

    dataframe['Normalized Score'] = scaler_model.transform(dataframe[['Score']])  
    dataframe=dataframe.reset_index(drop=True)
    return dataframe


def run_ml_script(df2):
    # Data Preprocessing to feed the model

    df2['Revenue (US Dollars)'] = np.where(df2['Revenue (US Dollars)'] == np.nan, df2['Annual Revenue'],
                                           df2['Revenue (US Dollars)'])
    df2['Revenue (US Dollars)'] = np.where(df2['Revenue (US Dollars)'] == 0, df2['Annual Revenue'],
                                           df2['Revenue (US Dollars)'])
    df2['Revenue (US Dollars)'] = df2['Revenue (US Dollars)'].fillna(0)

    revenueList = []

    # replace $ or , in Revenue (US Dollars) value
    for value in df2['Revenue (US Dollars)']:
        currVal = clean_currency(value)
        revenueList.append(currVal)

    df2['Revenue (US Dollars)'] = revenueList
    print(f"__Revenue (US Dollars): {df2['Revenue (US Dollars)']}")

    df2.dropna().drop_duplicates()

    final_df = df2.drop_duplicates()

    final_df['Employee Count Total'] = final_df['Employee Count Total'].replace(
        '', np.nan, regex=True)
    final_df['Employee Count Total'] = final_df['Employee Count Total'].fillna(
        0)
    print(f"__Employee_Count_Total {final_df['Employee Count Total']}")

    # print("year started", final_df['Year Started'])
    final_df['Year Started'] = final_df['Year Started'].replace(
        '', np.nan, regex=True).fillna(0)

    final_df['Year Started'] = final_df['Year Started'].astype('int')

    print("__Year_Started", "\n", final_df['Year Started'])

    print("__Last_Update_Date", "\n", final_df['Last Update Date'])

    # Null Hypothesis for a good Customer Partner(H0)

    # #1.The Annual Revenue must be high
    #
    # #2.The cp must be old in terms of age
    #
    # #3.The must updated recently
    #
    # #4.No of employee must be at higher side

    # s = "20120213"
    # # you could also import date instead of datetime and use that.
    # date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))

    # Replace missing values in 'Last Update Date' column with NaN
    final_df['Last Update Date'] = final_df['Last Update Date'].replace(
        '', np.nan, regex=True)

    # Convert non-missing date strings to datetime objects
    final_df['Last Update Date'] = pd.to_datetime(
        final_df['Last Update Date'], format='%Y-%m-%d', errors='coerce')

    # Replace missing datetime objects with the current date
    final_df['Last Update Date'] = final_df['Last Update Date'].fillna(
        datetime.now())

    # Add a day_since column showing the difference between last purchase and a basedate
    basedate = datetime.now()
    baseyear = datetime.now().year
    final_df['Recency'] = (basedate - final_df['Last Update Date']).dt.days
    final_df['Year Started'] = np.where(((final_df['Year Started'] < 1800) | (final_df['Year Started'] > baseyear)), baseyear,
                                        final_df['Year Started'])

    # Find the age of the company

    # Convert 'Year Started' column to datetime objects
    start_dates = pd.to_datetime(final_df['Year Started'], format='%Y')

    # Calculate the age of each company in days
    ages = (pd.Timestamp(datetime.now().date()) - start_dates).dt.days
    # Assign the ages to a new 'Age' column in the DataFrame
    final_df['Age'] = ages

    final_df['BEMFAB (Marketability)'] = np.where(
        final_df['BEMFAB (Marketability)'].isin(['Marketable', 'marketable', True]), 1, 0)

    # # Model Prediction

    # Load the saved pickle file generated from training.py file

    with open('model.pkl', 'rb') as load_file:
        model = pickle.load(load_file)

    with open('quantiles.pkl', 'rb') as load_file_1:
        quantiles = pickle.load(load_file_1)

    print("<<<Model loading Sucessful>>>")

    # Assigning weatage

    # We create five classes for the RFM segmentation since, being high revenue is good, high Employee Count Total is good ,High recency is bad and high age is good for bussiness entity.

    # model_df = final_df[
    #     ['Account Name', 'Revenue (US Dollars)', 'Employee Count Total', 'Recency', 'Age', 'BEMFAB (Marketability)']]
    model_df = final_df[['zohoAccountId', 'D-U-N-S Number', 'Account Name', 'Revenue (US Dollars)', 'Employee Count Total',
                         'Recency', 'Age', 'BEMFAB (Marketability)', 'rating', 'user_ratings_total', 'business_status', 'googlePlaceJson', 'dnbOptimzerJson']]

    # # Creating function to give weightage accordance with +ve and -ve corr
    #
    # We create five classes for the RFM segmentation since, being high revenue is good, high Employee Count Total is good ,High recency is bad and high age is good for bussiness entity.

    # For positive correlation

    def RClass(x, p, d):
        if float(x) <= d[p][0.25]:
            return 1
        elif float(x) <= d[p][0.50]:
            return 2
        elif float(x) <= d[p][0.75]:
            return 3
        else:
            return 4

    # For negetive correlation
    def FMClass(x, p, d):
        if float(x) <= d[p][0.25]:
            return 4
        elif float(x) <= d[p][0.50]:
            return 3
        elif float(x) <= d[p][0.75]:
            return 2
        else:
            return 1

    # For positive corerlation
    model_df['Revenue (US Dollars)'] = model_df['Revenue (US Dollars)'].replace(
        '', np.nan, regex=True).fillna(0)
    model_df['Rev_Quartile'] = model_df['Revenue (US Dollars)'].apply(
        RClass, args=('Revenue (US Dollars)', quantiles,))
    model_df['Emp_Quartile'] = model_df['Employee Count Total'].apply(
        RClass, args=('Employee Count Total', quantiles,))
    model_df['Age_Quartile'] = model_df['Age'].apply(
        RClass, args=('Age', quantiles,))

    # For negetive corerlation
    model_df['Rncy_Quartile'] = model_df['Recency'].apply(
        FMClass, args=('Recency', quantiles,))

    model_df['Confidence Indicator'] = (model_df['BEMFAB (Marketability)'] + model_df['Rev_Quartile'] +
                                        model_df['Emp_Quartile'] + model_df['Age_Quartile'] + model_df[
                                            'Rncy_Quartile']) / 17

    # # Method-2 (K-Mean Clustering)

    # Accoring to the elbow method we are safe to say that there are 3 cluster forming.

    X_test = model_df[['BEMFAB (Marketability)', 'Rev_Quartile',
                       'Emp_Quartile', 'Age_Quartile', 'Rncy_Quartile']]

    y_predict = model.predict(X_test)

    model_df['Cluster'] = y_predict
    model_df['riskLevel'] = 1

    # ###From the above value we have the idea that
    #
    # 'LOW' ==> (Class 2)
    #
    # 'MEDIUM' ==> (Class 1)
    #
    # 'HIGH' ==> (Class 0)

    model_df['riskLevel'] = np.where(
        model_df['Cluster'] == 2, 'LOW', model_df['riskLevel'])
    model_df['riskLevel'] = np.where(
        model_df['Cluster'] == 1, 'MEDIUM', model_df['riskLevel'])
    model_df['riskLevel'] = np.where(
        model_df['Cluster'] == 0, 'HIGH', model_df['riskLevel'])

    today_date = datetime.now().date()

    model_df['Date'] = today_date

    saved_data = model_df.drop('Cluster', axis=1)

    saved_data['Lowest'] = saved_data[['Rev_Quartile',
                                       'Emp_Quartile', 'Age_Quartile', 'Rncy_Quartile']].idxmin(axis=1)

    # Call the update safe class function to handel the corner caes

    saved_data["riskLevel"] = saved_data.apply(update_safe_class_1, axis=1)
    saved_data["riskLevel"] = saved_data.apply(update_safe_class_2, axis=1)
    saved_data["riskLevel"] = saved_data.apply(update_safe_class_3, axis=1)

    # Adding the cluster scor to the model

    new_score_df=saved_data[['Revenue (US Dollars)', 'Employee Count Total',
       'Recency', 'Age', 'BEMFAB (Marketability)','riskLevel']]
    
    saved_data=saved_data.reset_index(drop=True)


    try:
        saved_data['cluster_score'] = (calculate_score(new_score_df)['Normalized Score']).round(2)
    except Exception as e:
        saved_data['cluster_score'] = np.nan
        logger.exception("Error while calculating cluster score: %s", e)


    saved_data['Lowest'] = saved_data[['Rev_Quartile',
                                       'Emp_Quartile', 'Age_Quartile', 'Rncy_Quartile']].idxmin(axis=1)
    
    #Adding remark column to the result

    saved_data['Remark'] = np.nan
    saved_data.loc[(
        ((saved_data['riskLevel'] == 'HIGH') | (saved_data['riskLevel'] == 'MEDIUM')) & (
            saved_data['Lowest'] == 'Rev_Quartile')), 'Remark'] = 'Low Revenue'
    saved_data.loc[(
        ((saved_data['riskLevel'] == 'HIGH') | (saved_data['riskLevel'] == 'MEDIUM')) & (
            saved_data['Lowest'] == 'Emp_Quartile')), 'Remark'] = 'Low Employee Count'
    saved_data.loc[(
        ((saved_data['riskLevel'] == 'HIGH') | (saved_data['riskLevel'] == 'MEDIUM')) & (
            saved_data['Lowest'] == 'Age_Quartile')), 'Remark'] = 'Low Age Company'
    saved_data.loc[(
        ((saved_data['riskLevel'] == 'HIGH') | (saved_data['riskLevel'] == 'MEDIUM')) & (
            saved_data['Lowest'] == 'Rncy_Quartile')), 'Remark'] = 'Low Recency'

    saved_data = saved_data.drop('Lowest', axis=1)
    print(saved_data[['Age','Recency']])

    print("Final Result Generated>>>>>")
    return saved_data
