import requests

url = "http://localhost:5000/predict"

data = {
    "id": "80000",
    "city_code" : "C7",
    "region_code" : "3125",
    "accomodation_type" : "Owned",
    "reco_insurance_type" : "Individual", 
    "upper_age" : "66",
    "lower_age" : "36",
    "is_spouse" : "No",
    "health_indicator" : "X2",
    "holding_policy_duration" : "1.0",
    "holding_policy_type" : "3.0",
    "reco_policy_cat" : "22",
    "reco_policy_premium" : "17192.0"
 }

response = requests.post(url, json=data).json()
print(response)

# if response["prediction"] == True:
#     print(f"The lead {data["id"]} is a potential customer.")
# else:
#     print(f"The lead {data["id"]} is not a potential customer.")
