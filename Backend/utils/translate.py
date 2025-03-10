from translate import Translator
from langdetect import detect
from utils.trainingModel import input_question
from utils.datasets import add_datasets
from utils.fetchData import  callData

def translateEN(text):
        check = detect(text)
        print(text)
        if check == "vi":
            translator= Translator(to_lang="en",from_lang="vi")
            translation = translator.translate(f"{text}")
            return translation
        elif check == "en":
            return text


def translateVN(text):
    Translate = Translator(to_lang="vi",from_lang="en")
    translatetion = Translate.translate(f"{text}")
    print(translatetion)
    return translatetion