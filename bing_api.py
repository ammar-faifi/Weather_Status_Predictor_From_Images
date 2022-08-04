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


# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
endpoint = "https://api.bing.microsoft.com/v7.0/images/search"

# Query term(s) to search for.
queries = ["weather sunny", "weather foggy"]


# Create a diroctory for each query, if not exist
for query in queries:
    if not query in os.listdir():
        os.mkdir(query)


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
                f"{query}/{i}.{img['encodingFormat']}", "wb"
            ) as file:
                file.write(res.content)

    except Exception as ex:
        raise ex
