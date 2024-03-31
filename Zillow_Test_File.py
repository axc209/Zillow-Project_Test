import aiohttp
import pandas as pd


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

# Drop all columns with blanks
Clean_Details = normalized_details.dropna(axis=1, how="all")
Clean_Details.dtypes


drop_list = [
    "big",
    "hdpUrl",
    "hiResImageLink",
    "hugePhotos",
    "mapTileGoogleMapUrlFullWidthLightbox",
    "mediumImageLink",
    "nearbyCities",
    "nearbyHomes",
    "nearbyNeighborhoods",
    "nearbyZipcodes",
    "openHouseSchedule",
    "pals",
    "photos",
    "postingUrl",
    "responsivePhotos",
    "responsivePhotosOriginalRatio",
    "streetViewMetadataUrlMapLightboxAddress",
    "streetViewMetadataUrlMediaWallAddress",
    "streetViewMetadataUrlMediaWallLatLong",
    "streetViewServiceUrl",
    "streetViewTileImageUrlMediumAddress",
    "streetViewTileImageUrlMediumLatLong",
    "thumb",
    "tourPhotos",
    "virtualTourUrl",
    "apartmentsForRentInZipcodeSearchUrl.path",
    "attributionInfo.infoString10",
    "attributionInfo.listingAgents",
    "attributionInfo.listingOffices",
    "attributionInfo.mlsDisclaimer",
    "citySearchUrl.path",
    "formattedChip.location",
    "homeRecommendations.blendedRecs",
    "housesForRentInZipcodeSearchUrl.path",
    "listing_agent.image_data.url",
    "listing_agent.profile_url",
    "listing_agent.reviews_url",
    "listing_agent.services_offered",
    "listing_agent.username",
    "listing_agent.write_review_url",
    "onsiteMessage.eventId",
    "onsiteMessage.messages",
    "resoFacts.view",
    "resoFacts.virtualTour",
    "resoFacts.virtualTourURLUnbranded",
    "stateSearchUrl.path",
    "staticMap.sources",
    "streetView.addressSources",
    "streetView.latLongSources",
    "thirdPartyVirtualTour.externalUrl",
    "thirdPartyVirtualTour.lightboxUrl",
    "thirdPartyVirtualTour.providerKey",
    "thirdPartyVirtualTour.providerName",
    "thirdPartyVirtualTour.staticUrl",
    "tourEligibility.isPropertyTourEligible",
    "vrModel.cdnHost",
    "vrModel.revisionId",
    "vrModel.vrModelGuid",
    "zipcodeSearchUrl.path",
    "neighborhoodSearchUrl.path",
    "listing_agent.phone",
    "listingProvider.logos",
    "listingProvider.postingWebsiteURL",
    "listingProvider.sourceText",
    "listingProvider.title",
    "communityUrl.path",
    "richMedia.floorPlan",
    "richMedia.imx",
    "richMedia.virtualTour",
    "foreclosureMoreInfo.apn",
    "foreclosureMoreInfo.legalDescription",
    "foreclosureMoreInfo.recordedDocs",
]
drop_list_1 = [
    "desktopWebHdpImageLink",
    "floorMaps",
    "mapTileGoogleMapUrlFullWidthLightboxWholeZipcode",
    "mapTileGoogleMapUrlFullWidthMax",
    "mapTileGoogleMapUrlFullWidthMaxWholeZipcode",
    "mapTileGoogleMapUrlSmall",
    "mapTileGoogleMapUrlSmallWholeZipcode",
]
Clean_Details.drop(columns=drop_list, inplace=True)
Clean_Details.drop(columns=drop_list_1, inplace=True)
Clean_Details.to_excel("Zillow_Details.xlsx", index=False)

# Keep columns
keep = [
    "ZPID",
    "abbreviatedAddress",
    "bathrooms",
    "bedrooms",
    "brokerIdDimension",
    "brokerageName",
    "buildingId",
    "city",
    "country",
    "county",
    "datePostedString",
    "datePriceChanged",
    "dateSold",
    "dateSoldString",
    "daysOnZillow",
    "description",
    "favoriteCount",
    "foreclosureAmount",
    "hasVRModel",
    "hdpTypeDimension",
    "homeInsights",
    "homeStatus",
    "homeType",
    "isListedByOwner",
    "isPremierBuilder",
    "isRentalListingOffMarket",
    "keystoneHomeStatus",
    "latitude",
    "listPriceLow",
    "livingArea",
    "livingAreaUnits",
    "livingAreaUnitsShort",
    "longitude",
    "lotSize",
    "monthlyHoaFee",
    "moveInReady",
    "pageViewCount",
    "photoCount",
    "postingProductType",
    "price",
    "priceChange",
    "priceChangeDate",
    "priceChangeDateString",
    "priceHistory",
    "propertyTaxRate",
    "propertyTypeDimension",
    "regionString",
    "rentZestimate",
    "rentalApplicationsAcceptedType",
    "restimateHighPercent",
    "restimateLowPercent",
    "schools",
    "sellingSoon",
    "taxAssessedValue",
    "taxAssessedYear",
    "taxHistory",
    "timeZone",
    "tourViewCount",
    "yearBuilt",
    "zestimate",
    "zestimateHighPercent",
    "zestimateLowPercent",
    "address.subdivision",
    "downPaymentAssistance.maxAssistance",
    "downPaymentAssistance.resultCount",
    "foreclosureTypes.isBankOwned",
    "listing_agent.display_name",
    "listing_agent.first_name",
    "listing_agent.rating_average",
    "listing_agent.recent_sales",
    "listing_agent.review_count",
    "mortgageRates.arm5Rate",
    "mortgageRates.fifteenYearFixedRate",
    "mortgageRates.thirtyYearFixedRate",
    "resoFacts.accessibilityFeatures",
    "resoFacts.appliances",
    "resoFacts.architecturalStyle",
    "resoFacts.associationAmenities",
    "resoFacts.associationFee",
    "resoFacts.associationFeeIncludes",
    "resoFacts.atAGlanceFacts",
    "resoFacts.availabilityDate",
    "resoFacts.basement",
    "resoFacts.basementYN",
    "resoFacts.bathrooms",
    "resoFacts.bathroomsFull",
    "resoFacts.bathroomsHalf",
    "resoFacts.bathroomsOneQuarter",
    "resoFacts.bathroomsThreeQuarter",
    "resoFacts.bedrooms",
    "resoFacts.builderModel",
    "resoFacts.builderName",
    "resoFacts.buildingArea",
    "resoFacts.buildingAreaSource",
    "resoFacts.buyerAgencyCompensation",
    "resoFacts.buyerAgencyCompensationType",
    "resoFacts.canRaiseHorses",
    "resoFacts.carportSpaces",
    "resoFacts.cityRegion",
    "resoFacts.communityFeatures",
    "resoFacts.constructionMaterials",
    "resoFacts.cooling",
    "resoFacts.coveredSpaces",
    "resoFacts.daysOnZillow",
    "resoFacts.depositsAndFees",
    "resoFacts.electric",
    "resoFacts.elementarySchool",
    "resoFacts.elementarySchoolDistrict",
    "resoFacts.entryLocation",
    "resoFacts.exclusions",
    "resoFacts.exteriorFeatures",
    "resoFacts.fencing",
    "resoFacts.fireplaceFeatures",
    "resoFacts.fireplaces",
    "resoFacts.flooring",
    "resoFacts.foundationDetails",
    "resoFacts.furnished",
    "resoFacts.garageSpaces",
    "resoFacts.greenEnergyEfficient",
    "resoFacts.greenIndoorAirQuality",
    "resoFacts.greenSustainability",
    "resoFacts.greenWaterConservation",
    "resoFacts.hasAdditionalParcels",
    "resoFacts.hasAssociation",
    "resoFacts.hasAttachedGarage",
    "resoFacts.hasAttachedProperty",
    "resoFacts.hasCarport",
    "resoFacts.hasCooling",
    "resoFacts.hasFireplace",
    "resoFacts.hasGarage",
    "resoFacts.hasHeating",
    "resoFacts.hasHomeWarranty",
    "resoFacts.hasLandLease",
    "resoFacts.hasOpenParking",
    "resoFacts.hasPrivatePool",
    "resoFacts.hasSpa",
    "resoFacts.hasView",
    "resoFacts.hasWaterfrontView",
    "resoFacts.heating",
    "resoFacts.highSchool",
    "resoFacts.highSchoolDistrict",
    "resoFacts.hoaFee",
    "resoFacts.homeType",
    "resoFacts.horseYN",
    "resoFacts.interiorFeatures",
    "resoFacts.isNewConstruction",
    "resoFacts.isSeniorCommunity",
    "resoFacts.laundryFeatures",
    "resoFacts.leaseTerm",
    "resoFacts.levels",
    "resoFacts.listingTerms",
    "resoFacts.livingArea",
    "resoFacts.livingAreaRangeUnits",
    "resoFacts.lotFeatures",
    "resoFacts.lotSize",
    "resoFacts.lotSizeDimensions",
    "resoFacts.middleOrJuniorSchool",
    "resoFacts.middleOrJuniorSchoolDistrict",
    "resoFacts.onMarketDate",
    "resoFacts.openParkingSpaces",
    "resoFacts.otherEquipment",
    "resoFacts.otherFacts",
    "resoFacts.otherParking",
    "resoFacts.otherStructures",
    "resoFacts.ownership",
    "resoFacts.parcelNumber",
    "resoFacts.parking",
    "resoFacts.parkingFeatures",
    "resoFacts.patioAndPorchFeatures",
    "resoFacts.poolFeatures",
    "resoFacts.pricePerSquareFoot",
    "resoFacts.propertyCondition",
    "resoFacts.propertySubType",
    "resoFacts.roadSurfaceType",
    "resoFacts.roofType",
    "resoFacts.rooms",
    "resoFacts.securityFeatures",
    "resoFacts.sewer",
    "resoFacts.spaFeatures",
    "resoFacts.specialListingConditions",
    "resoFacts.stories",
    "resoFacts.storiesTotal",
    "resoFacts.structureType",
    "resoFacts.subdivisionName",
    "resoFacts.taxAnnualAmount",
    "resoFacts.topography",
    "resoFacts.utilities",
    "resoFacts.vegetation",
    "resoFacts.waterBodyName",
    "resoFacts.waterSource",
    "resoFacts.waterView",
    "resoFacts.waterViewYN",
    "resoFacts.waterfrontFeatures",
    "resoFacts.windowFeatures",
    "resoFacts.yearBuilt",
    "resoFacts.yearBuiltEffective",
]

Clean_Details = Clean_Details[keep]
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


# Specify the columns and their types (dict or list of dicts)
columns_to_expand = {
    "homeInsights": "dict",
    "priceHistory": "list",
    "schools": "list",
    "sellingSoon": "dict",
    "taxHistory": "list",
    "resoFacts.atAGlanceFacts": "list",
    "resoFacts.rooms": "list",
}

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


# List of columns that contain lists to be separated into individual columns
list_columns = [
    "resoFacts.accessibilityFeatures",
    "resoFacts.appliances",
    "resoFacts.communityFeatures",
    "resoFacts.constructionMaterials",
    "resoFacts.cooling",
    "resoFacts.exclusions",
    "resoFacts.exteriorFeatures",
    "resoFacts.fireplaceFeatures",
    "resoFacts.flooring",
    "resoFacts.foundationDetails",
    "resoFacts.greenEnergyEfficient",
    "resoFacts.greenIndoorAirQuality",
    "resoFacts.greenSustainability",
    "resoFacts.greenWaterConservation",
    "resoFacts.heating",
    "resoFacts.interiorFeatures",
    "resoFacts.laundryFeatures",
    "resoFacts.lotFeatures",
    "resoFacts.otherEquipment",
    "resoFacts.otherStructures",
    "resoFacts.parkingFeatures",
    "resoFacts.patioAndPorchFeatures",
    "resoFacts.poolFeatures",
    "resoFacts.propertySubType",
    "resoFacts.roadSurfaceType",
    "resoFacts.securityFeatures",
    "resoFacts.sewer",
    "resoFacts.spaFeatures",
    "resoFacts.utilities",
    "resoFacts.vegetation",
    "resoFacts.waterSource",
    "resoFacts.waterfrontFeatures",
    "resoFacts.windowFeatures",
]

# Initialize an empty DataFrame to store the binary columns
binary_columns_df = pd.DataFrame(index=final_clean.index)

# Process each column that contains a list
for col in list_columns:
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
final_clean = final_clean.drop(columns=list_columns).join(binary_columns_df)


drop_list_2 = [
    "buildingId",
    "description",
    "hdpTypeDimension",
    "homeStatus",
    "livingAreaUnits",
    "propertyTypeDimension",
    "regionString",
    "resoFacts.otherFacts",
    "resoFacts.parcelNumber",
    "homeInsights_0",
    "priceHistory_attributeSource",
    "priceHistory_buyerAgent",
    "priceHistory_sellerAgent",
    "schools_link",
    "resoFacts.atAGlanceFacts_factValue",
]
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
