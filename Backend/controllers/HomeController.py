import os
from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)
import urllib3
http = urllib3.PoolManager()
import json
import asyncio
import random
from langdetect import detect
from fastapi import HTTPException
from utils.translate import translateEN, translateVN
from utils.datasets import add_datasets
from utils.fetchData import  callData
from utils.trainingModel import input_question



class HomeController:
    @staticmethod
    def test():
        botData ={"question": "can you training me about marketing some quession: anwser ?",} 
        try:
            with open("./datasets/Marketing.json", "r") as file:
                data = json.load(file) 
                rn = random.randint(1,2)
                done = callData(botData["question"],rn)
                t = done.split("\n")
                a = []
                result =""
               
                print(t)
    
                return {"message":f"{t}"}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=400, detail=str(e))  
      

    @staticmethod
    def input(item): 
        body = item.input.strip() 
        try:
            answer = input_question(body)
            if answer == "Sorry, I don't understand your question.":
                swapEN = translateEN(body)
                rn = random.randint(1,2)
                data = callData(swapEN,rn)
                ch_1 = detect(body)
                print(ch_1)
                print(swapEN)
                add_datasets(swapEN,data)
                print("1")
               
                if ch_1 =="vi":
                    print("2")
                    tran = translateVN(data)
                    return {"message":f"{tran}","quesion": body}
                else:
                    return {"message":f"{data}","quesion": body}
            else:
                print("3")
                ch_2 = detect(answer)
                if ch_2 == "en":
                    print("4")
                    return {"message":f"{answer}","quesion": body}
                else:
                    tran1 = translateVN(answer)
                    return {"message":f"{tran1}","quesion":body}
            return {"message":f"{answer}","quesion":body}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=400, detail=str(e))



