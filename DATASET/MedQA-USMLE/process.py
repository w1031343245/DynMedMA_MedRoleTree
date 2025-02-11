import json

with open("/home/user1/WR/Auto_Method/Dataset/MedQA-USMLE/train/train_10178.json", 'r', encoding="utf-8") as f:
    data = json.load(f)

convert_data = []

for instance in data["instances"]:
    da = instance["text"]

    question = str(instance).split("Input: ", 1)[1]

    answer = question.split("Output: ", 1)[1]
    question = question.split("Output: ", 1)[0]

    option_A = question.split("(A) ", 1)[1]
    question = question.split("(A) ", 1)[0]
    option_B = option_A.split("(B) ", 1)[1]
    option_A = option_A.split("(B) ", 1)[0]
    option_C = option_B.split("(C) ", 1)[1]
    option_B = option_B.split("(C) ", 1)[0]
    option_D = option_C.split("(D) ", 1)[1]
    option_C = option_C.split("(D) ", 1)[0]
    answer = answer.split(" ", 1)[0]

    if(answer == "A"): context = option_A
    elif(answer == "B"): context = option_B
    elif(answer == "C"): context = option_C
    else:context = option_D

    convert_data.append({
        "question" :question,
        "answer_idx" : answer,
        "options": {"A": option_A, "B": option_B, "C": option_C, "D": option_D},
        "answer": context,
        "meta_info": "example"
    })

with open("/home/user1/WR/MED_ROLE_TREE/DATASET/MedQA-USMLE/MedQA_train.jsonl", "w", encoding = "utf-8") as f:
    for data in convert_data:
        json.dump(data, f, ensure_ascii = False)
        f.write("\n")

