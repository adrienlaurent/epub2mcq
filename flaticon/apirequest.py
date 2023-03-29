import requests
import json
import pprint

params = { 'apikey' : '' }

response = requests.post("https://api.flaticon.com/v2/app/authentication", params=params)


print(response.reason)
print(response.status_code)
print(response.text)

j = json.loads(response.text)

#print(json.dumps(j['data']['token'], indent=4))

headers = {
  'Accept':'application/json',
  'Authorization': 'Bearer ' + j['data']['token'],
}


params = { 
    'limit' : '2',
    'q' : 'hoghouse',
}

#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(params)


response2 = requests.get("https://api.flaticon.com/v2/search/icons/priority", headers=headers, params=params)


#print(response2.reason)
#print(response2.status_code)

j2 = json.loads(response2.text)

print(json.dumps(j2, indent=4))
