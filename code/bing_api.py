"""
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
"""
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import requests
from  pathlib import Path

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
endpoint = "https://api.bing.microsoft.com/v7.0/images/search"

# Query term(s) to search for.
classes = {
    "sunny": ["sunny weather", "sunny sky"],
    "foggy": ["foggy weather", "foggy day"],
    "cloudy": ["cloudy weather", "cloudy day"],
    "rainy": ["rainy weather", "rainy day"],
    "snowy": ["snowy weather", "snowy day"],
}


# Create a diroctory for each query, if not exist
DATA_FOLDER = "data"
BASE_DIR = Path(__file__).parent
DATA_FOLDER = BASE_DIR / DATA_FOLDER

for class_ in classes:
    # create folder for each cat
    if not class_ in os.listdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER / class_)


for class_, queries in classes.items():
    for query in queries:
        # Construct a request
        mkt = "en-US"
        params = {"q": query, "mkt": mkt, "count": 150}
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}

        # Call the API
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            with open("bing_search.json", "w", encoding="utf-8") as file:
                file.write(json.dumps(data))

            for i, img in enumerate(data["value"]):
                print(img["thumbnailUrl"], img["encodingFormat"])

                res = requests.get(img["thumbnailUrl"], stream=True)
                with open(
                    f"{DATA_FOLDER}/{class_}/{query}-{i}.{img['encodingFormat']}", "wb"
                ) as file:
                    file.write(res.content)

        except Exception as ex:
            raise ex
