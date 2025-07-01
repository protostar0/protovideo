import requests

API_KEY = "51130176-384192dfe2b7414fdaaa3a322"
url = "https://pixabay.com/api/music/"
params = {
    "key": API_KEY,
    "q": "trading",         # search keyword
    "genre": "corporate",   # optional
    "per_page": 3
}

response = requests.get(url, params=params, proxies={"http":"http://127.0.0.1:8080","https":"http://127.0.0.1:8080"},verify=False)
data = response.json()

for hit in data["hits"]:
    print("Title:", hit["title"])
    print("URL:", hit["audio"])
    print("Duration:", hit["duration"], "seconds")
    print("Download link:", hit["audio"])
