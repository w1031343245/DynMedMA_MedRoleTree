import json
from umls_python_client import UMLSClient
import time

def check_from_umls(word: str):
    api_key = "b1599216-80f9-47b4-ad87-d4fe308e188c"
    search_api = UMLSClient(api_key=api_key).searchAPI

    search_results = search_api.search(
        search_string=word
    )
    time.sleep(0.1)
    print(f"Search Results for 'diabetes': {search_results}")
    if isinstance(search_results, str):
        data_ini = json.loads(search_results)
    else:
        return None, None
    if 'result' in data_ini:
        data_ini = data_ini['result']
    else:
        return None, None
    
    if len(data_ini['results']): 
        data_ini = data_ini['results'][0]
        if len(data_ini['name']) < 50:
            return data_ini['ui'], data_ini['name']
        else: return None, None
    else: return None, None
