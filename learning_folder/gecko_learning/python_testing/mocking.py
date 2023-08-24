import requests
import json


def send_request_to_get_users():
    response = requests.get(url='https://jsonplaceholder.typicode.com/users')
    return response.content


def convert_to_json_readable(content):
    ret = json.loads(content.decode())
    return [item for item in ret if item['id']<5]
