import asyncio
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from os.path import join, dirname
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
keyGemini = os.environ.get("GEMINI")



def callData(item,rn):
    data =[]
    try:
        if rn == 1:
            url = "http://localhost:11434/api/generate"
            payload = json.dumps({
            "model": "llama3.2",
            "prompt": f"{item}"
            })
            headers = {
            'Content-Type': 'application/json'
            }
            response = requests.post(url, headers=headers, data=payload)
            
            res = response.text
            json_objects = []
            for line in res.split("\n"):
                try:
                    json_objects.append(json.loads(line))  
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} on line: {line}")
            kp = ""
            for result in json_objects:
                kp += result.get("response")

            return kp
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={keyGemini}"
            headers= {"Content-Type": "application/json"}
            payload = json.dumps({       
                "contents": [{
                    "parts":[{"text": f"{item}"}]
                    }]
                })
            response = requests.post(url, headers=headers, data=payload)
            json_objects = []
            res = response.json()
            check = res["candidates"]
            for i in check:
                for y in i["content"]["parts"]:
                    json_objects.append(y["text"])
            kp = ""
            for result in json_objects:
                kp += result
            return kp        
            
    except Exception as e:
        print("Error", e)
        print(str(e))


