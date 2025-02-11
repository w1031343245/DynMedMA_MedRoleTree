from umls_python_client import UMLSClient
from LABEL_DEFINE.Label import *
from UMLS_USE.ui_to_standard import *
from EMBEDDING.embedding import *
import json
import os
import time
import torch

def is_exit(exist_word: list, ui: str):
    flag = "good"
    for word in exist_word:
        if word == ui: flag = "bad"

    return flag

def build_family(word:str, cui: str, father: str, son: list, exist_word: list, output_dir: str, label_list: list[data_label], deepth: int):
    if not os.path.exists(f"{output_dir}/embedding"): os.makedirs(f"{output_dir}/embedding", exist_ok=True)
    if deepth == 2: return exist_word, label_list 
    ff = 1
    if_have_father = 0
    if len(son) == 0 and father == "": ff = 0
    exist_word.append(cui)
    father_node: str = father
    son_nodes = son
    API_KEY  = "b1599216-80f9-47b4-ad87-d4fe308e188c"
    cui_api = UMLSClient(api_key=API_KEY).cuiAPI
    relation_info = cui_api.get_relations(cui=cui)
    time.sleep(0.1)
    if isinstance(relation_info, str):
        datas = json.loads(relation_info)
    else:
        return exist_word, label_list
    if 'result' in datas:
        relation_datas = datas['result']
    else:
        relation_datas = []

    for data in relation_datas:
        if data['relationLabel'] == "CHD" and ff == 0 and 'relatedIdName' in data and if_have_father == 0: 
            print(f"该节点是{word}节点的父亲子节点{data['relatedIdName']}")
            father_data_ui, father_data_name = check_from_umls({data['relatedIdName']})
            if father_data_name != data['relatedIdName']:
                print(f"该单词{data['relatedIdName']}已经进行标准化变成单词{father_data_name}！！！！")
            if father_data_ui == None and father_data_name == None or len(father_data_name) >= 30:
                print("该节点不存在！！！！")
                continue
            flag = is_exit(exist_word, father_data_ui)
            if flag == "good":
                print(f"{father_data_name}并没有存入{word}的家族树中，下面进行查找该单词！！！！！！")
                father_node = father_data_name
                exist_word, label_list  = build_family(father_data_name, father_data_ui, "", [word], exist_word, output_dir, label_list, 0)
                if_have_father = 1
            else: 
                print(f"{data['relatedIdName']}已经存入{word}的家族树中！！！！！！")
        if data['relationLabel'] == "PAR" and 'relatedIdName' in data and deepth < 1: 
            print(f"该节点是{word}节点的儿子节点{data['relatedIdName']}")
            son_data_ui, son_data_name = check_from_umls(data['relatedIdName'])
            if son_data_name != data['relatedIdName']:
                print(f"该单词{data['relatedIdName']}已经进行标准化变成单词{son_data_name}！！！！")
            if son_data_ui == None and son_data_name == None or len(son_data_name) >= 30:
                print("该节点不存在！！！！")
                continue
            flag = is_exit(exist_word, son_data_ui)
            if flag == "good": 
                print(f"{son_data_name}并没有存入{word}的家族树中，下面进行查找该单词！！！！！！")
                son_nodes.append(son_data_name)
                exist_word, label_list = build_family(son_data_name, son_data_ui, word, [], exist_word, output_dir, label_list, deepth=deepth + 1)
            else: print(f"{son_data_name}已经存入{word}的家族树中！！！！！！")
        
    print(f"{word}的父子节点已经找完！！！！！")
    label_node = data_label(cui, word)
    label_node.build_word_to_vector(embedding(word, f"{output_dir}/embedding"))
    time.sleep(0.5)
    label_node.build_father(father_node)
    label_node.build_son(son_nodes)
    label_list.append(label_node)
    # data_dict = {
    #     "ui": label_node.ui,
    #     "name": label_node.name,
    #     "father_label": label_node.father_label,
    #     "son_label": label_node.son_label,
    #     "word_to_vector": label_node.word_to_vector
    # }

    
    # if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok = True)
    # with open(f"{output_dir}/label_tree_data.jsonl", "a", newline="") as f:
    #     json.dump(data_dict, f)
    #     f.write("\n")

    return exist_word, label_list


