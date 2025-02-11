import numpy as np
from LABEL_DEFINE.Label import *
from CLUSTER_WITH_LMM.prompt import *
from UMLS_USE.ui_to_standard import *
from LLM_API.qwen_7b_api import *
from LLM_API.gpt4_api import *
from LLM_API.gpt35_api import *
from EMBEDDING.embedding import *
from UMLS_USE.build_family import *
import re
from scipy.spatial.distance import cdist

def extract_cluster_label(text):
    match = re.search(r'Cluster label:\s*(.*)', text)
    if match:
        return match.group(1).strip() 
    else:
        return None

def cluster_word(word1: str, word2: str, n: int):
    print(word1, word2)
    new_cluster = LLM_CLUSTER_GPT(word1, word2)
    context = chat_with_gpt(new_cluster)
    # if n <= 15: context = chat_with_gpt(new_cluster, 0.5, 200, 0.5)
    # else: context = chat_with_qwen(new_cluster)
    print(context)
    new_label = extract_cluster_label(context).lower()
    print(new_label)
    return new_label, n + 1

def cluster_word_plus(word1: str, word2: str, fail_label:list, n: int):
    new_cluster = LLM_CLUSTER_plus_GPT(word1, word2, fail_label)
    print(new_cluster)
    context = chat_with_gpt(new_cluster)
    # if n <= 15: context = chat_with_gpt(new_cluster, 0.5, 200, 0.5)
    # else: context = chat_with_qwen(new_cluster)
    print(context)
    new_label = extract_cluster_label(context).lower()
    print(new_label)
    return new_label, n + 1

def is_exit(exist_word: list, ui: str):
    flag = "good"
    for word in exist_word:
        if word == ui: flag = "bad"

    return flag

def data_logging(label_list:list[data_label], model:str, num: int, source: str):
    label_family_dir = f"/home/user1/WR/MED_ROLE_TREE/RESULT/{source}/Final_Label_Tree/{model}/{num}"
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
    
def add_new_label(son_word1:str, son_word2:str, new_father_word: str, new_father_cui: str,label_list: list[data_label], exist_words: list, output_dir: str):
    for i in range(len(label_list)):
        if label_list[i].name == son_word1 or label_list[i].name == son_word2:
            label_list[i].build_father(new_father_word)
            print(f"节点{label_list[i].name}已添加父节点{new_father_word}！！！！！")
    son_nodes = []
    son_nodes.append(son_word1)
    son_nodes.append(son_word2)
    print(f"新父节点{new_father_word}使用UMLS进行构建深度为2的子节点构建！！！！")
    exist_words, label_list = build_family(new_father_word, new_father_cui, "", son_nodes, exist_words, output_dir, label_list, 0)

    return exist_words, label_list
        
def collect_tree_nodes(root, parent_map, collected = None):
    if collected is None:
        collected = set()
    for child in parent_map.get(root, []):
        collect_tree_nodes(child, parent_map, collected)
    return collected

def cluster(embedding_list: list, word_list: list, exist_words: list ,label_list: list[data_label], model: str, num: int, source: str):
    LFT_output_dir = f"/home/user1/WR/MED_ROLE_TREE/RESULT/{source}/Label_Family_Tree/{model}/{num}"
    Final_Label_Tree = f"/home/user1/WR/MED_ROLE_TREE/RESULT/{source}/Final_Label_Tree/{model}/{num}"
    dist_matrix = cdist(embedding_list, embedding_list, "euclidean")

    np.fill_diagonal(dist_matrix, np.inf)

    while len(word_list) > 1:
        n = 0
        print(f"当前没有父节点数量还有{len(word_list)}")
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        print(word_list[i], word_list[j])

        fail_label = []
        new_label, n = cluster_word(word_list[i], word_list[j], n)
        print(f"新生成的父节点是{new_label}！！！！！")
        while new_label == None or len(new_label) >= 50:
            print("新的父节点不合格！！！！！！")
            new_label, n = cluster_word(word_list[i], word_list[j], n)
            print(f"新生成的父节点是{new_label}！！！！！")
        combine_label_ui, combine_label_name = check_from_umls(new_label)
        print(combine_label_ui, combine_label_name)
        while combine_label_ui == None and combine_label_name == None or new_label == None or len(new_label) >= 50 or is_exit(exist_words, combine_label_ui) == "bad":
            print("新的父节点不合格！！！！！！")
            print(combine_label_ui, combine_label_name)
            if new_label != None: fail_label.append(new_label)
            fail_label = list(set(fail_label))
            new_label, n = cluster_word_plus(word_list[i], word_list[j], fail_label, n)
            if new_label == None: continue
            combine_label_ui, combine_label_name = check_from_umls(new_label)
            print(f"新生成的父节点是{new_label}！！！！！")
        print("1")
        # if is_exit(exist_words, combine_label_ui) == "bad":
        #     if combine_label_name == word_list[i]:
        #         print(f"{word_list[j]}作为子节点存入{word_list[i]}！！！！")
        #         for k in range(len(label_list)):
        #             if label_list[k].name == word_list[i]:
        #                 label_list[k].son_label.append(word_list[j])
        #             if label_list[k].name == word_list[j]:
        #                 label_list[k].father_label = word_list[i]
        #         word_list = [word_list[k] for k in range(len(word_list)) if k != j]
        #         embedding_list = [embedding_list[k] for k in range(len(embedding_list)) if k != j]
        #         dist_matrix = np.delete(dist_matrix, [j], axis=0)
        #         dist_matrix = np.delete(dist_matrix, [j], axis=1)
        #     elif combine_label_name == word_list[j]:
        #         print(f"{word_list[i]}作为子节点存入{word_list[j]}！！！！")
        #         for k in range(len(label_list)):
        #             if label_list[k].name == word_list[j]:
        #                 label_list[k].son_label.append(word_list[i])
        #             if label_list[k].name == word_list[i]:
        #                 label_list[k].father_label = word_list[j]
        #         word_list = [word_list[k] for k in range(len(word_list)) if k !=i]
        #         embedding_list = [embedding_list[k] for k in range(len(embedding_list)) if k != i]
        #         dist_matrix = np.delete(dist_matrix, [i], axis=0)
        #         dist_matrix = np.delete(dist_matrix, [i], axis=1)
        #     else:
            #     print(f"{word_list[j]}和{word_list[i]}作为子节点存入{combine_label_name}！！！！")
            #     for k in range(len(label_list)):
            #         if label_list[k].name == combine_label_name:
            #             label_list[k].son_label.append(word_list[i])
            #             label_list[k].son_label.append(word_list[j])
            #         if label_list[k].name == word_list[i] or label_list[k].name == word_list[j]:
            #             label_list[k].build_father(combine_label_name)
            #     word_list = [word_list[k] for k in range(len(word_list)) if k !=i and k != j]
            #     embedding_list = [embedding_list[k] for k in range(len(embedding_list)) if k != i and k != j]
            #     dist_matrix = np.delete(dist_matrix, [i, j], axis=0)
            #     dist_matrix = np.delete(dist_matrix, [i, j], axis=1)
            # data_logging(label_list, model, num, source)            
        # else:
        odir = embedding(combine_label_name, f"{LFT_output_dir}/embedding")
        new_label_embedding = np.load(odir)
        print(exist_words)
        print(combine_label_ui)
        exist_words.append(combine_label_ui)
        new_index = len(word_list)
        print(f"新的父节点{combine_label_name}生成成功并经过检测！！！！！！")
        dist_matrix = np.pad(dist_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=np.inf)
        for k in range(dist_matrix.shape[0] - 1):
            if k != i and k != j:
                dist_matrix[k, new_index] = np.linalg.norm(embedding_list[k] - new_label_embedding)
                dist_matrix[new_index, k] = dist_matrix[k, new_index]

        dist_matrix = np.delete(dist_matrix, [i, j], axis=0)
        dist_matrix = np.delete(dist_matrix, [i, j], axis=1)
        print(f"新的聚合标签存入数列的步骤！！！")
        exist_words, label_list = add_new_label(word_list[i], word_list[j], combine_label_name, combine_label_ui, label_list, exist_words, Final_Label_Tree)
        word_list = [word_list[k] for k in range(len(word_list)) if k !=i and k != j]
        embedding_list = [embedding_list[k] for k in range(len(embedding_list)) if k != i and k != j]
        embedding_list.append(new_label_embedding)
        word_list.append(combine_label_name)
        data_logging(label_list, model, num, source)

    return exist_words, label_list

        


