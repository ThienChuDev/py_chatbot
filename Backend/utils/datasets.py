import json

def add_datasets(quesion,answer):
    new_data = {"question": f"{quesion}", "answer": f"{answer}"}
    with open("./datasets/Marketing.json", "r") as file:
        data = json.load(file)
        data.append(new_data)    
    with open("./datasets/Marketing.json", "w") as file:
      json.dump(data, file,indent=2, ensure_ascii=False)
    return True


