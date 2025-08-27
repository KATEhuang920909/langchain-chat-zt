from langchain.utilities import BingSearchAPIWrapper
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_SUBSCRIPTION_KEY = '6e17dcbd02504082be552e9144dbd961'
search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,bing_search_url=BING_SEARCH_URL)
print(search.results("唱跳rap篮球的是谁？", 3))
print(1)