import aiohttp
import pandas as pd
import requests


async def fetch_data(session, location, page):
    url = "https://zillow56.p.rapidapi.com/search"
    querystring = {
        "location": location,
        "status": "forSale",
        "isSingleFamily": "true",
        "isMultiFamily": "true",
        "isApartment": "false",
        "isCondo": "false",
        "isManufactured": "false",
        "isTownhouse": "false",
        "isLotLand": "false",
        "page": page,
    }
    headers = {
        "X-RapidAPI-Key": "INSERT API KEY",
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com",
    }
    async with session.get(url, headers=headers, params=querystring) as response:
        if response.status == 200:
            data = await response.json()
            print(f"Page {page} fetched for {location}")
            return data
        print(f"Failed to fetch page {page} for {location}")
        return None


async def main(locations):
    async with aiohttp.ClientSession() as session:
        all_data = []
        for location in locations:
            page = 1
            total_pages = 1  # Initialize with a value to ensure at least one iteration
            location_data = []

            while page <= total_pages:
                data = await fetch_data(session, location, page)
                if data and "results" in data and data["results"]:
                    location_data.extend(data["results"])
                    total_pages = data.get("totalPages", 1)
                    page += 1
                else:
                    print(f"No more data or error fetching {location} page {page}")
                    break

            all_data.extend(location_data)
            print(f"Finished fetching data for {location}")

        df = pd.DataFrame(all_data)
        return df


locations = [
    "dallas, tx",
    "irving, tx",
    "garland, tx",
    "grand prairie, tx",
    "mesquite, tx",
    "carrolton, tx",
    "richardson, tx",
    "rowlett, tx",
    "plano, tx",
    "addison, tx",
    "arlington, tx",
]
df = await main(locations)


df.dtypes
import ast


# Function to safely evaluate a string into a dictionary
def safe_eval_dict(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return None


# Convert 'open_house_info' column from string representation of dictionaries to actual dictionaries
df["open_house_info"] = df["open_house_info"].apply(safe_eval_dict)

# Extract first open house showing start and end times
df["first_open_house_start"] = df["open_house_info"].apply(
    lambda x: (
        x.get("open_house_showing")[0]["open_house_start"]
        if x is not None
        and "open_house_showing" in x
        and len(x["open_house_showing"]) > 0
        else None
    )
)
df["first_open_house_end"] = df["open_house_info"].apply(
    lambda x: (
        x.get("open_house_showing")[0]["open_house_end"]
        if x is not None
        and "open_house_showing" in x
        and len(x["open_house_showing"]) > 0
        else None
    )
)

# Display the first few rows to verify the new columns
df[["open_house_info", "first_open_house_start", "first_open_house_end"]].head()

# Convert 'listing_sub_type' column from string representations of dictionaries to actual dictionaries
df["listing_sub_type"] = df["listing_sub_type"].apply(safe_eval_dict)

# Dynamically create columns for each key in the 'listing_sub_type' dictionaries
for row in df["listing_sub_type"].dropna():
    for key in row.keys():
        df[key] = df["listing_sub_type"].apply(
            lambda x: x.get(key) if x is not None else None
        )

# Extract numerical values from 'priceReduction' column
df["priceReduction"] = (
    df["priceReduction"]
    .str.extract(r"[\$]([\d,]+)")
    .replace(",", "", regex=True)
    .astype(float)
)

df.dtypes
df.to_excel("Zillow.xlsx", index=False)
# Make zpid list
zpid_list = df["zpid"].tolist()
zpid_list = [str(zpid) for zpid in zpid_list]


"""


######################################
MORE PROPERTY DETAILS
######################################


"""


# details = response.json()
# dets = pd.json_normalize(details)


# Initialize an empty list to store the results
dets = []

# Loop through each ZPID in the list
for zpid in zpid_list:
    # Update the query string with the current ZPID
    querystring = {"zpid": zpid}

    # Make the API request
    response = requests.get(
        "https://zillow56.p.rapidapi.com/property", headers=headers, params=querystring
    )

    # Check if the request was successful
    if response.status_code == 200:
        # Add the response JSON to the results list, including the ZPID
        details = response.json()  # Assuming this is a dictionary
        # Include the ZPID with the data
        dets_with_zpid = {"ZPID": zpid, "Data": details}
        dets.append(dets_with_zpid)
    else:
        print(f"Failed to get data for ZPID {zpid}")

detail_results = pd.DataFrame(dets)

normalized_details = pd.DataFrame()
for index, row in detail_results.iterrows():
    temp_df_1 = pd.json_normalize(row["Data"])
    temp_df_1["ZPID"] = row["ZPID"]
    normalized_details = pd.concat([normalized_details, temp_df_1], ignore_index=True)


column = normalized_details.pop("ZPID")
normalized_details.insert(0, "ZPID", column)

# normalized_details.to_excel('zillow_details.xlsx',index=False)

"""

#####################################
CLEAN PROPERTY DETAILS
#####################################


"""

from columns_config import (
    keep_columns,
    drop_list,
    drop_list_1,
    columnns_containing_lists,
    columns_to_expand,
    drop_list_2,
)

# Drop all columns with blanks
Clean_Details = normalized_details.dropna(axis=1, how="all")
Clean_Details.dtypes

Clean_Details.drop(columns=drop_list, inplace=True)
Clean_Details.drop(columns=drop_list_1, inplace=True)
Clean_Details.to_excel("Zillow_Details.xlsx", index=False)

Clean_Details = Clean_Details[keep_columns]
# Clean_Details.to_excel('Zillow_Details.xlsx',index=False)


# Function to expand dictionaries directly into columns
def expand_dict_col(Clean_Details, col_name):
    # Check if the column exists and is not empty
    if col_name in Clean_Details and not Clean_Details[col_name].isnull().all():
        # Normalize and expand the dictionary into a DataFrame
        expanded_col = Clean_Details[col_name].apply(pd.Series)
        # Rename columns to reflect their origin
        expanded_col.columns = [
            f"{col_name}_{sub_col}" for sub_col in expanded_col.columns
        ]
        return expanded_col
    else:
        return pd.DataFrame()


# Function to handle columns with lists of dictionaries by expanding the first dictionary
def expand_list_of_dicts_col(Clean_Details, col_name):
    if col_name in Clean_Details and not Clean_Details[col_name].isnull().all():
        # Normalize and expand the first dictionary of the list into a DataFrame
        expanded_col = (
            Clean_Details[col_name]
            .apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {})
            .apply(pd.Series)
        )
        # Rename columns to reflect their origin
        expanded_col.columns = [
            f"{col_name}_{sub_col}" for sub_col in expanded_col.columns
        ]
        return expanded_col
    else:
        return pd.DataFrame()


# Initialize an empty DataFrame to hold all expanded columns
expanded_data = pd.DataFrame(index=df.index)

# Process each specified column based on its type
for col, col_type in columns_to_expand.items():
    if col_type == "dict":
        expanded_data = expanded_data.join(expand_dict_col(Clean_Details, col))
    elif col_type == "list":
        expanded_data = expanded_data.join(expand_list_of_dicts_col(Clean_Details, col))

# Combine expanded data back into the original DataFrame, excluding original complex-structure columns
final_clean = Clean_Details.drop(columns=list(columns_to_expand.keys())).join(
    expanded_data
)

final_clean.to_excel("Zillow_Details.xlsx", index=False)

# Drop all columns with blanks
final_clean = final_clean.dropna(axis=1, how="all")


# Define the safe_eval function to handle potential errors and mixed types
def safe_eval(val):
    try:
        # Only evaluate if val is a string
        if isinstance(val, str):
            return ast.literal_eval(val)
        # If val is not a string (e.g., already a list), return it directly
        return val
    except (ValueError, SyntaxError):
        # Return None or some default value in case of error
        return None


# Initialize an empty DataFrame to store the binary columns
binary_columns_df = pd.DataFrame(index=final_clean.index)

# Process each column that contains a list
for col in columnns_containing_lists:
    # Check if column exists to avoid KeyErrors
    final_clean[col] = final_clean[col].apply(safe_eval)
    if col in final_clean.columns:
        # The line below is removed because safe_eval already handles the conversion
        # exploded_df = final_clean[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else []).explode()

        # Explode the DataFrame to have each list item in its row, while keeping the index for aggregation
        exploded_df = final_clean[col].explode()

        # Get dummy variables for each item in the list, then group by index and sum to aggregate the binary representation
        dummies = pd.get_dummies(exploded_df).groupby(level=0).sum()

        # Prefix the new columns with the original column name for clarity
        dummies.columns = [f"{col}_{dummy}" for dummy in dummies.columns]

        # Join the dummies back to the binary_columns_df
        binary_columns_df = binary_columns_df.join(dummies, how="outer")

# Now, concatenate the binary columns DataFrame with the original df, dropping the original list columns for simplicity
final_clean = final_clean.drop(columns=columnns_containing_lists).join(
    binary_columns_df
)

final_clean.drop(columns=drop_list_2, inplace=True)

# Use str.extract() with a regular expression to separate numbers and text
# The pattern (\d+) matches one or more digits (the fee amount)
# The pattern ([^\d]+) matches one or more non-digit characters (the text part)
final_clean[["associationFee_amount", "associationFee_text"]] = final_clean[
    "resoFacts.associationFee"
].str.extract(r"(\d+)([^\d]+)")

# Convert the extracted amount to numeric type, if necessary
final_clean["associationFee_amount"] = pd.to_numeric(
    final_clean["associationFee_amount"], errors="coerce"
)

# List of columns to convert
columns_seconds_to_date = [
    "datePriceChanged",
    "dateSold",
    "priceChangeDate",
    "resoFacts.onMarketDate",
]

for col in columns_seconds_to_date:
    if col in final_clean.columns:
        # Convert seconds to datetime, assuming the original time is in UTC
        final_clean[col] = pd.to_datetime(final_clean[col] / 1000, unit="s", utc=True)

        # Convert from UTC to Central Time ('America/Chicago')
        final_clean[col] = final_clean[col].dt.tz_convert("America/Chicago")

# Error exporting to excel
# ValueError: Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.
# Have to output excel files for Stephanie
columns_to_modify = [
    "datePriceChanged",
    "dateSold",
    "priceChangeDate",
    "resoFacts.onMarketDate",
]

for col in columns_to_modify:
    if col in final_clean.columns:
        # Convert the datetime objects to timezone-naive in Central Time
        final_clean[col] = final_clean[col].dt.tz_localize(None)

final_clean.to_excel("Zillow_Details.xlsx", index=False)
