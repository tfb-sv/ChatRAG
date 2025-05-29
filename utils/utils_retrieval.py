import time
import requests

# Load SerpAPI credentials and base URL
SERP_TOKEN = open("utils/credentials_serp.env", encoding="utf-8").read().strip()
SERP_URL = "https://serpapi.com/search"

NUM_SEARCHES = 20

# Perform Google search using SerpAPI
def search_query(query, lang):
    start_time = time.time()

    # Set up search parameters
    url = SERP_URL
    params = {
        "q": query,
        "api_key": SERP_TOKEN,
        "engine": "google",
        "num": NUM_SEARCHES,
        "hl": lang,
        "gl": lang,
    }

    # Send GET request
    res = requests.get(url, params=params)
    data = res.json()

    # Extract title + snippet from results
    results = []
    for result in data.get("organic_results", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        if snippet:
            results.append(f"{title}. {snippet}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"    > Search time: {elapsed:.2f} seconds")

    return results

# Example static search result for testing
def example_search_result(lang):
    if lang == "tr":
        results_raw = {
          "organic_results": [
            {
              "title": "2024 İhracat Raporu",
              "snippet": "Türkiye'nin 2024 yılı ihracatı 261,9 milyar dolar olarak gerçekleşti."
            },
            {
              "title": "TÜİK Verileri",
              "snippet": "2024'te ihracat %2,5 artışla 261,9 milyar dolar oldu."
            }
          ]
        }
    elif lang == "en":
        results_raw = {
          "organic_results": [
            {
              "title": "2024 Export Report",
              "snippet": "Turkey's exports in 2024 amounted to 261.9 billion dollars."
            },
            {
              "title": "TURKSTAT Data",
              "snippet": "In 2024, exports increased by 2.5% to 261.9 billion dollars."
            }
          ]
        }

    # Extract and format search results
    results = []
    for item in results_raw["organic_results"]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if snippet:
            results.append(f"{title}. {snippet}")

    return results
