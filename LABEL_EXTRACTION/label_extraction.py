import tqdm
import csv
from DATASET.data import *
from LLM_API.qwen_7b_api import *
from LLM_API.gpt35_api import *
from LLM_API.gpt4_api import *
from LABEL_EXTRACTION.prompt import *
from LABEL_EXTRACTION.data_clean import *
from UMLS_USE.ui_to_standard import *
from UMLS_USE.build_family import *

def data_logging(label_list:list[data_label], model:str, num: int, source: str, data_name: str):
    label_family_dir = f"/home/user1/WR/MED_ROLE_TREE/RESULT_{data_name}/{source}/Label_Family_Tree/{model}/{num}"
    if not os.path.exists(label_family_dir): os.makedirs(label_family_dir, exist_ok=True)
    
    with open(f"{label_family_dir}/label_tree_data.jsonl", "w", newline="") as f:
        for data in label_list:
            data_dict = {
                "ui": data.ui,
                "name": data.name,
                "father_label": data.father_label,
                "son_label": data.son_label,
                "word_to_vector": data.word_to_vector
            }
            json.dump(data_dict, f)
            f.write("\n")

def is_exit(exist_words: list, ui: str):
    flag = "good"
    for word in exist_words:
        if word == ui: flag = "bad"

    return flag

def KE(num: int, model: str, exist_words: list, label_list: list[data_label], begin_num: int, sum_l: list, source: str, data_name: str):
    sum: int
    sum_list = sum_l
    output_dir = f"/home/user1/WR/MED_ROLE_TREE/RESULT_{data_name}/{source}/Label_Family_Tree/{model}/{num}"
    data = MyDataset(source, data_name)
    test_range = num
    for idx in tqdm.tqdm(range(begin_num, test_range), desc = f"{begin_num + 1} ~ {test_range}"):
        print(f"当前是第{idx}道问题！！！！！")
        raw_sample = data.get_by_idx(idx)

        if data_name == "MEDICAL_BOOK":
            con = raw_sample['paragraph']
        else:
            question = raw_sample["question"]
            options = raw_sample["options"]
        if data_name == "PubMedQA":
            context = raw_sample["context"]
            input_context = str(question) + str(context)
        elif data_name == "MEDICAL_BOOK":
            input_context = str(con)
        else:
            input_context = str(question) + str(options)
         
        
        finish_words = []
        j = 0
        for i in range(4):
            if(model == "gpt3.5"):
                if i == 0: Ke = DOMAIN_EXTRACTION_GPT(input_context)
                else: Ke = DOMAIN_EXTRACTION_GPT_REPEAT(input_context, finish_words)
                context = chat_with_gpt(Ke)
            elif model == "gpt4":
                if i == 0: Ke = DOMAIN_EXTRACTION_GPT(input_context)
                else: Ke = DOMAIN_EXTRACTION_GPT_REPEAT(input_context, finish_words)
                context = chat_with_gpt4(Ke)
            else:
                if i == 0: Ke = DOMAIN_EXTRACTION_GPT(input_context)
                else: Ke = DOMAIN_EXTRACTION_GPT_REPEAT(input_context, finish_words)
                context = chat_with_qwen(Ke, 0, 500, 1)

            print(context)
     
            question_K = context.split(":")[-1].strip().split(", ")
            label_ini = question_K[0]
            label_ini = data_clean(label_ini)
            cui, label = check_from_umls(label_ini)
            finish_words.append(label_ini)
            if cui != None and label != None and is_exit(exist_words, cui) != "bad" and len(label) < 30:
                print(f"{label_ini}经过质量检测，标准化为单词{label}进入家族树的构建！！！！！")
                exist_words,label_list = build_family(label, cui, "", [], exist_words, output_dir, label_list, 1)
                i = i - 1
            else: 
                print(f"{label_ini}质量不合格！！！！")
                j += 1
                i = i - 1
            if j >= 2: break
        sum = len(exist_words)
        sum_list.append(sum)
        print(f"当前是第{idx}道问题，累计抽取的知识点为{sum}！！！！！")
        data_logging(label_list, model, num, source, data_name)
        print(f"当前是第{idx}道问题，已经将所有的标签存入jsonl文件！！！！！")
        print(sum_list)
        with open(f"/home/user1/WR/MED_ROLE_TREE/RESULT_{data_name}/{source}/Label_Family_Tree/{model}/{num}/sum_list.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in sum_list:
                writer.writerow([item])
    print(exist_words)
    print(len(exist_words))
    print(sum_list)

    

    return exist_words, label_list

