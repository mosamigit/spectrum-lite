#!/usr/bin/env python
# coding: utf-8

# OLD Logic
# Reads data from an Excel spreadsheet (in the same directory)
# Data related to CP - based on DnB Optimizer
# Prepare data frame to process by the ML code
# ML code process generates Scorecard
# riskLevel, riskNotes
# Write Scorecard data to Excel Spreadsheet

# CURRENT Logic
# EXPECTED changes
# Read Zoho Deals data - #findZohoDealRecords API
# criteria: Stage, Type, Closing_Date
# Fetch CP > DnB Optimizer data - #fetchDnBOptimizerBizByZohoAccountIds API
# Search & Fetch Google rating data - #SearchGooglePlaces & fetchGooglePlaceByPlaceId API
# Prepare data frame to process by the ML code
# ML code process generates Scorecard
# riskLevel, riskNotes
# Push Scorecard data (via #upsertMlScorecard API) to Zoho

import os
# import re
import json
# Import all the required libraries
from datetime import datetime
import pytz
import sys

import re
# import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.structures import CaseInsensitiveDict
from glom import glom
from ml import run_ml_script
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

deal_stage = "Closed Won,Closed Won (Logistics)"
deal_type = "New Business"
scored_by = "SPECTRUM_LITE"

if "DEAL_STAGE_VALUE" in os.environ:
    deal_stage = os.getenv("DEAL_STAGE_VALUE")


def spectrum_lite_run():
    def get_api_key_token(api_client_key, api_client_secret, url_endpoint):
        try:
            payload = generate_payload_api_token_payload()
            print(f"Requesting access token for CLIENT_ID - {api_client_key}")
            response = requests.post(url_endpoint, json=payload, auth=(
                api_client_key, api_client_secret))
            response.raise_for_status()
            response = response.json()

            if "errors" in response:
                error_message = response["errors"][0]["message"]
                print(
                    f"Error occurred while fetching API key token: {error_message}")
                return None
            else:
                return response["data"]["apiKeyToken"]["token"]

        except (requests.exceptions.HTTPError, KeyError, ValueError) as e:
            print(f"Error occurred while fetching API key token: {e}")
            return None

    def generate_payload_api_token_payload():
        query = """mutation apiKeyToken($grant_type: String!) 
                {apiKeyToken(grant_type: $grant_type)
                {
                token
                issuedAt 
                expiresAt }}"""
        variables = {"grant_type": "client_credentials"}
        payload = {"query": query, "variables": variables}

        return payload

    def get_all_deals_data(bearer_token, endpoint):

        url = endpoint
        payload = findZohoDealRecordsPayload()
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        response = requests.post(url, json=payload, headers=headers)
        response = response.json()
        return response

    def findZohoDealRecordsPayload():
        # Convert to America/Los_Angeles - PST - time zone
        dateNow = datetime.now(tz=pytz.timezone('America/Los_Angeles'))
        print("==== #findZohoDealRecordsPayload ==== ")
        print(f"dateNow: {dateNow}")
        print(f"deal_closing_date initial value based on today's date: {dateNow}")
        end_date = dateNow.strftime("%Y-%m-%d")
        start_date = dateNow.replace(day=1).strftime("%Y-%m-%d")
        deal_closing_date=f"{start_date},{end_date}"

        # if "DEAL_CLOSING_DATE_VALUE" in os.environ:
        #     deal_closing_date = os.getenv("DEAL_CLOSING_DATE_VALUE")

        date_criteria_operator = "between"
        if "DEAL_CLOSING_DATE_OPERATOR" in os.environ:
            date_criteria_operator = os.getenv("DEAL_CLOSING_DATE_OPERATOR")

        print(f"deal_stage: {deal_stage}")
        print(f"deal_type: {deal_type}")
        print(f"date_criteria_operator: {date_criteria_operator}")
        print(f"deal_closing_date: {deal_closing_date}")

        query = """ query {
                findZohoDealRecords(
                    criteria: [
                        {
                            fieldName: "Stage",
                            operator: in,
                            value: "%s"
                        },
                        {
                            fieldName: "Type",
                            operator: equals,
                            value: "%s"
                        },
                        {
                            fieldName: "Closing_Date",
                            operator: %s,
                            value: "%s"
                        }
                    ]
                ) {
                id
                Deal_Name
                Stage
                Closing_Date
                Account_Name {
                    id
                    Account_Name
                }
            }
            } """ % (deal_stage, deal_type, date_criteria_operator, deal_closing_date)
        deals_query = {"query": query, "variables": {}}
        # print()
        # print(deals_query)
        print("==== ========= ==== ")
        return deals_query

    def get_all_accounts_data(bearer_token, endpoint, account_ids):
        url = endpoint
        query = """ query {
                fetchDnBOptimizerBizByZohoAccountIds(
                    zohoAccountIds: %s
                ) {
                    zohoAccountId
                    data
                }
            } """ % account_ids

        accounts_query = {"query": query, "variables": {}}
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        response = requests.post(url, json=accounts_query, headers=headers)
        response = response.json()
        return response

    def get_DnB_data_byDunsNumber(bearer_token, endpoint, dunsNumber):
        url = endpoint
        query = """ query GetAssessmentByDunsNumber {
                        GetAssessmentByDunsNumber(
                            dunsNumber: "%s", 
                            searchDB: false
                        ) {
                            id
                            businessName
                            dunsNumber
                            fullResponse
                        }
                    } """ % (dunsNumber)

        dnb_query = {"query": query, "variables": {}}
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        response = requests.post(url, json=dnb_query, headers=headers)
        response = response.json()
        return response

    def get_google_place_data(bearer_token, endpoint, name, code, city, state, country):
        url = endpoint
        query = """ query SearchGooglePlaces {
                        searchGooglePlaces(
                            where: {
                                name: "%s",
                                postalCode: "%s",
                                city: "%s",
                                state: "%s",
                                country: "%s"
                            }
                        ) {
                            place_id
                            name
                            formatted_address
                        }
                    } """ % (name, code, city, state, country)

        google_place_query = {"query": query, "variables": {}}
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        response = requests.post(url, json=google_place_query, headers=headers)
        response = response.json()
        return response

    def get_google_place_rating(bearer_token, endpoint, google_place_id):
        url = endpoint
        query = """ query FetchGooglePlaceByPlaceId {
        fetchGooglePlaceByPlaceId(
            place_id: "%s"
        ) {
            id
            place_id
            website
            rating
            user_ratings_total
            business_status
            zohoAccountId
        }
    } """ % google_place_id

        google_place_query = {"query": query, "variables": {}}
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        response = requests.post(url, json=google_place_query, headers=headers)
        response = response.json()
        return response
    
    def check_if_scorecard_needs_upsert(bearer_token, endpoint, duns_number, risk_level):
        payload = get_payload_to_check_if_record_exists(duns_number)
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        response = requests.post(endpoint, json=payload, headers=headers)
        print("Scorecard Pulled Data====================")
        response = response.json()['data']
        print(response)
        print("====================")
        response = response["mlScorecard"]
            
        if (response is not None):
            existing_risk_level = response["riskLevel"]
            print("Existing risk level:", existing_risk_level)
            if existing_risk_level != risk_level:
                upsertInput = {"risk_level": risk_level}
                print(f"#upsetMlScorecard - {upsertInput}")
                return upsertInput
            else:
                print("No upsert required as riskLevel is not changed !")
                return None
        else:
            upsertInput = {"risk_level": risk_level}
            return upsertInput
            
    def get_payload_to_check_if_record_exists(duns_number):
        query = """query mlScorecard {
        mlScorecard(where:{
            dunsNumber_scoredBy: {
                dunsNumber: "%s"
                scoredBy: SPECTRUM_LITE
            }
        }) {
            id
            createdAt
            updatedAt
            dunsNumber
            myBizId
            zohoAccountId
            riskLevel
            riskScore
            paymentTerms
            creditLimit
            scoredBy
            business {
                id
                name
                legalName
                myBizId
            }
        }
        }""" % duns_number
        payload = {"query": query, "variables": {}}
        return payload

    def upsert_spectrum_lite_score(bearer_token, endpoint, duns_number, zoho_account_id, risk_level, cluster_score, dnbOptimzerJson,
                                   spectrumLiteOutputJson, googlePlaceJson):

        upsertInput = {"duns_number": duns_number, "scored_by": scored_by,
                       "risk_level": risk_level, "cluster_score": cluster_score, "zoho_account_id": zoho_account_id}
        print(f"#upsetMlScorecard - {upsertInput}")

        if (duns_number == '' or duns_number == None):
            print(
                f"  - Skipping #upsertMlScorecard call as the D-U-N-S Number not found for the zohoAccountId#{record['zohoAccountId']}")
            return
        
        result = check_if_scorecard_needs_upsert(bearer_token, endpoint, duns_number, risk_level)
        if result is None:
            return

        url = endpoint
        query = """mutation upsertMlScorecard($data: MlScorecardCreateUpdateInput!) {
                          upsertMlScorecard(
                            data: $data
                          ) {
                            id
                            scoredBy
                            dunsNumber
                            riskLevel
                            clusterScore
                            zohoAccountId
                          }
                        } """
        variables = {
            "data": {
                "dunsNumber": duns_number,
                "scoredBy": scored_by,
                "riskLevel": risk_level,
                "clusterScore": cluster_score,
                "zohoAccountId": zoho_account_id,
                "dnbOptimzerJson": json.dumps(dnbOptimzerJson),
                "googlePlaceJson": json.dumps(googlePlaceJson),
                "spectrumLiteOutputJson": json.dumps(spectrumLiteOutputJson)
            }
        }
        upset_query = {"query": query, "variables": variables}
        headers = CaseInsensitiveDict()
        headers["Authorization"] = "Bearer %s" % bearer_token
        # === code for testing
        # response = """{ "dunsNumber": "%s", "scoredBy": "%s", "riskLevel": "%s", "clusterScore": "%s", "zohoAccountId": "%s" }""" % (
        #     duns_number, scored_by, risk_level, cluster_score, zoho_account_id)
        # === ======
        response = requests.post(url, json=upset_query, headers=headers)
        response = response.json()
        print(f"  - result {response}")
        return response

    def del_none(value):
        """
        Recursively remove all None values from dictionaries and lists, and returns
        the result as a new dictionary or list.
        """
        if isinstance(value, list):
            return [del_none(x) for x in value if x is not None]
        elif isinstance(value, dict):
            return {
                key: del_none(val)
                for key, val in value.items()
                if val is not None
            }
        else:
            return value

    def findMax(arr, attrib):

        n = len(arr)
        max = 0
        if (n > 0):
            # Initialize maximum element
            max = arr[0][attrib]

            # Traverse array elements from second
            # and compare every element with
            # current max
            for i in range(1, n):
                if arr[i][attrib] > max:
                    max = arr[i][attrib]
        return max

    if "API_KEY_ML" not in os.environ or "API_KEY_ML" not in os.environ or "API_ENDPOINT_ML" not in os.environ:
        print("Environment file does not have mandatory filed values")

    if os.getenv("API_KEY_ML") == "" or os.getenv("API_KEY_ML") == "" or os.getenv("API_KEY_ML") == "":
        print("Environment file does not have mandatory field values")

    api_key = os.getenv("API_KEY_ML")
    api_secret = os.getenv("API_SECRET_ML")
    api_endpoint = os.getenv("API_ENDPOINT_ML")

    api_auth_token = get_api_key_token(api_key, api_secret, api_endpoint)

    if api_auth_token:
        # Make subsequent requests here using the API key token
        response = requests.get(api_endpoint, headers={
                                "Authorization": f"Bearer {api_auth_token}"})
        print(response)
        data_dict = get_all_deals_data(api_auth_token, api_endpoint)
    else:
        # Stop further execution of the script
        print("API key token not available. Unable to make subsequent requests.")
        sys.exit(1)

    if 'errors' in data_dict:
        print(data_dict["errors"])
    else:
        deals = data_dict["data"]
        # print(deals)
        deal_accounts = glom(
            deals, ("findZohoDealRecords", ["Account_Name"]))
        arr_account_ids = glom(deal_accounts, ["id"])
        dedup_arr_account_ids = list(dict.fromkeys(arr_account_ids))
        # dedup_arr_account_ids = ["4818701000005916016"]
        # print(f"dedup_arr_account_ids: {dedup_arr_account_ids}")

        if len(arr_account_ids) > 0:
            print("deals data fetched")
            print(
                f"zohoAccountIds based on the deals fetched: {json.dumps(dedup_arr_account_ids)}")
            accounts_data_dict = get_all_accounts_data(
                api_auth_token, api_endpoint, json.dumps(dedup_arr_account_ids))

            if 'errors' in accounts_data_dict:
                print(accounts_data_dict["errors"])
            else:
                print("Zoho Accounts data fetched")
                accounts = accounts_data_dict["data"]
                accounts_dict = glom(
                    accounts, ("fetchDnBOptimizerBizByZohoAccountIds", ["data"]))
                rem_none_accounts_dict = del_none(accounts_dict)

                # for debugging
                seperator = "|"
                print(
                    f"Zoho AccountId{seperator}D-U-N-S Number{seperator}Account Name")
                for i, record in enumerate(rem_none_accounts_dict):
                    debug_dunsNumber = ''
                    if 'D-U-N-S Number' in record and record['D-U-N-S Number'] is not None:
                        debug_dunsNumber = record['D-U-N-S Number']
                    print(
                        f"{record['zohoAccountId']}{seperator}{debug_dunsNumber}{seperator}{record['Account Name']}")

                # arr_google_api = []
                for i, record in enumerate(rem_none_accounts_dict):
                    user_rating = ""
                    user_rating_total = ""
                    business_status = ""

                    print("")
                    print(f"RECORD_dnbOptimizer: {record}")

                    if 'D-U-N-S Number' in record and record['D-U-N-S Number'] is not None:
                        # Fetch DnB data based on "D-U-N-S Number"
                        # This is to correctly receive the "D-U-N-S Number" with "0" prefix
                        print(
                            f"Fetching DnB data by DUNS number for - {record['Account Name']} {record['D-U-N-S Number']}")

                        dnbDataResponse = get_DnB_data_byDunsNumber(
                            api_auth_token, api_endpoint, record["D-U-N-S Number"])
                        # print(json.dumps(dnbDataResponse))
                        if 'errors' in dnbDataResponse:
                            print(
                                f"  - errors: {json.dumps(dnbDataResponse['errors'])}")
                        else:
                            dnbData = dnbDataResponse["data"]
                            dunsNumber = glom(
                                dnbData, ("GetAssessmentByDunsNumber", "dunsNumber"))
                            print(
                                f"DUNS number for - {record['Account Name']} {dunsNumber}")
                            record["D-U-N-S Number"] = dunsNumber

                            fullResponse = glom(
                                dnbData, ("GetAssessmentByDunsNumber", "fullResponse"))

                            if fullResponse and fullResponse.strip():
                                dnbFullResponse = json.loads(fullResponse)
                                yearStarted = glom(
                                    dnbFullResponse, ("organization", "startDate"))
                                lastUpdateDate = glom(
                                    dnbFullResponse, ("organization", "financingEvents", "financingStatementFilings", "mostRecentFilingDate"))
                                lastUpdateDate_1 = glom(
                                    dnbFullResponse, ("organization", "dunsControlStatus", "fullReportDate"))
                                numberOfEmployees = glom(
                                    dnbFullResponse, ("organization", "numberOfEmployees"))
                                employeeCountTotal = findMax(
                                    numberOfEmployees, "value")
                                bemfeb = glom(
                                    dnbFullResponse, ("organization", "dunsControlStatus", "isMarketable"))

                                # FIN-976 - use dnbData in lieu of DnB_Optimizer data
                                # Extract the year using a regular expression and the search() function
                                if yearStarted is not None:
                                    match = re.search(
                                        r"\d{4}", str(yearStarted))
                                    yearStarted = match.group()

                                record["Year Started"] = yearStarted
                                record["Employee Count Total"] = employeeCountTotal
                                record["BEMFAB (Marketability)"] = bemfeb

                                if pd.isna(lastUpdateDate) or lastUpdateDate is None or lastUpdateDate == 0:
                                    if pd.isna(lastUpdateDate_1) or lastUpdateDate_1 is None or lastUpdateDate_1 == 0:
                                        record["Last Update Date"] = datetime.today().strftime(
                                            '%Y-%m-%d')
                                    else:
                                        record["Last Update Date"] = lastUpdateDate_1
                                else:
                                    record["Last Update Date"] = lastUpdateDate

                                print(
                                    f"  DnB data - Year Started , Last Update Date, Employee Count Total, BEMFAB (Marketability) - {record['Year Started']}, {record['Last Update Date']}, {record['Employee Count Total']}, {record['BEMFAB (Marketability)']}")

                    # Fetch Google Place data based on CP name, city, state, postal code, country etc.
                    print(
                        f"Fetching Google Place data for - {record['Account Name']}")

                    if "Billing City" in record and "Billing State" in record and record["Billing City"] != "" and record["Billing State"] != "":

                        ginput_billing_code = ''
                        ginput_country = ''

                        if 'Billing Code' in record:
                            ginput_billing_code = record["Billing Code"]

                        if 'Billing Country' in record:
                            ginput_country = record["Billing Country"]

                        print(
                            f"  location details - {record['Billing City']} {record['Billing State']} {ginput_billing_code} {ginput_country}")

                        google_place = get_google_place_data(api_auth_token, api_endpoint, record["Account Name"],
                                                             ginput_billing_code, record["Billing City"],
                                                             record["Billing State"], ginput_country)
                        # print(json.dumps(google_place))
                        if 'errors' in google_place:
                            print(f"  - errors: {google_place['errors']}")
                            place_rating_data = google_place["errors"]
                        else:
                            place_data = google_place["data"]
                            place_id = glom(
                                place_data, ("searchGooglePlaces", ["place_id"]))
                            # print(place_id[0])
                            google_place_byId = get_google_place_rating(
                                api_auth_token, api_endpoint, place_id[0])

                            place_rating_data = google_place_byId["data"]
                            # print(place_rating_data)
                            if 'errors' in place_rating_data:
                                print(
                                    f"  - errors: {place_rating_data['errors']}")
                                place_rating_data = place_rating_data["errors"]
                            else:
                                user_rating = glom(
                                    place_rating_data, ("fetchGooglePlaceByPlaceId", "rating"))
                                user_rating_total = glom(place_rating_data, ("fetchGooglePlaceByPlaceId",
                                                                             "user_ratings_total"))
                                business_status = glom(
                                    place_rating_data, ("fetchGooglePlaceByPlaceId", "business_status"))
                                print("  - google place data fetched successfully")
                    else:
                        print(
                            "  - Cannot request as missing value for the required params - Billing City, Billing State")
                        place_rating_data = {}

                    rem_none_accounts_dict[i]["rating"] = user_rating
                    rem_none_accounts_dict[i]["user_ratings_total"] = user_rating_total
                    rem_none_accounts_dict[i]["business_status"] = business_status
                    rem_none_accounts_dict[i]["googlePlaceJson"] = json.dumps(
                        place_rating_data)
                    rem_none_accounts_dict[i]["dnbOptimzerJson"] = json.dumps(
                        record)

                # print(arr_google_api)
                df1 = pd.json_normalize(rem_none_accounts_dict)
                print("convert to data set")
                # df1.rename(columns={'zohoAccountId': 'Record Id'}, inplace=True)
                column_list = ['zohoAccountId', 'Account Name', 'Annual Revenue', 'Revenue (US Dollars)',
                               'D-U-N-S Number',
                               'Year Started', 'Last Update Date', 'Employee Count Total', 'BEMFAB (Marketability)',
                               'Billing Code',
                               'Billing State', 'Billing Country', 'Billing City', 'rating', 'user_ratings_total',
                               'business_status', 'googlePlaceJson', 'dnbOptimzerJson']

                if all(item in df1.columns for item in column_list):
                    print("checked the columns")
                    df2 = df1[column_list]
                    # print(df2)
                    result = run_ml_script(df2)
                    json_result = result.to_json(orient='records')
                    final_result = json.loads(json_result)

                    file_date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
                    file_name = 'output/spectrum_lite_' + file_date + '.json'
                    os.makedirs('output', exist_ok=True)

                    with open(file_name, 'w') as f:
                        f.write(json_result)

                    if len(final_result) > 0:
                        for record in final_result:
                            dnbOptimzerJson = json.loads(
                                record["dnbOptimzerJson"])
                            googlePlaceJson = json.loads(
                                record["googlePlaceJson"])

                            del dnbOptimzerJson['googlePlaceJson']
                            del record['dnbOptimzerJson']
                            del record['googlePlaceJson']

                            # record["D-U-N-S Number"]="117955689"

                            updated_result = upsert_spectrum_lite_score(
                                api_auth_token, api_endpoint, record["D-U-N-S Number"], record["zohoAccountId"], record["riskLevel"], record["cluster_score"], dnbOptimzerJson, record, googlePlaceJson)
                            # print("#upsetMlScorecard result - ", updated_result)

                else:
                    print("Columns are missing")
        else:
            print("Not deals found.")


if __name__ == '__main__':
    spectrum_lite_run()
