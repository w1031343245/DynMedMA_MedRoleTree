import numpy as np
from LABEL_DEFINE.Label import *
from CLUSTER_WITH_LMM.prompt import *
from UMLS_USE.ui_to_standard import *
from LLM_API.qwen_7b_api import *
from LLM_API.gpt4_api import *
from LLM_API.gpt35_api import *
from EMBEDDING.embedding import *
from UMLS_USE.build_family import *
from collections import defaultdict
import re
from scipy.spatial.distance import cdist

def extract_cluster_label(text):
    match = re.search(r'Cluster label:\s*(.*)', text)
    if match:
        return match.group(1).strip() 
    else:
        return None

def cluster_word(word1: str, word2: str, model: str, n: int):
    print(word1, word2)
    new_cluster = LLM_CLUSTER_GPT(word1, word2)
    if model == "gpt4" : context = chat_with_gpt4(new_cluster)
    elif model == "gpt3.5" or n >= 10: context = chat_with_gpt(new_cluster)
    elif model == "qwen2.5": context = chat_with_qwen(new_cluster, 0, 200, 1)
    # context = chat_with_gpt(new_cluster)
    while context == None or extract_cluster_label(context).lower() == None:
        if model == "gpt4" : context = chat_with_gpt4(new_cluster)
        elif model == "gpt3.5" or n >= 10: context = chat_with_gpt(new_cluster)
        elif model == "qwen2.5": context = chat_with_qwen(new_cluster, 0, 200, 1)
    print(context)
    new_label = extract_cluster_label(context).lower()
    print(new_label)
    return new_label, n + 1

def cluster_word_plus(word1: str, word2: str, fail_label:list, model: str, n: int):
    new_cluster = LLM_CLUSTER_plus_GPT(word1, word2, fail_label)
    print(new_cluster)
    if model == "gpt4" : context = chat_with_gpt4(new_cluster)
    elif model == "gpt3.5" or n >= 10: context = chat_with_gpt(new_cluster)
    elif model == "qwen2.5": context = chat_with_qwen(new_cluster, 0, 200, 1)
    # context = chat_with_gpt(new_cluster)
    while context == None or extract_cluster_label(context).lower() == None:
        if model == "gpt4" : context = chat_with_gpt4(new_cluster)
        elif model == "gpt3.5" or n >= 10: context = chat_with_gpt(new_cluster)
        elif model == "qwen2.5": context = chat_with_qwen(new_cluster, 0, 200, 1)
    print(context)
    new_label = extract_cluster_label(context).lower()
    print(new_label)
    return new_label, n + 1

def is_exit(exist_word: list, ui: str):
    flag = "good"
    for word in exist_word:
        if word == ui: flag = "bad"

    return flag

def data_logging(label_list:list[data_label], model:str, num: int, source: str, data_name: str):
    label_family_dir = f"/home/user1/WR/MED_ROLE_TREE/RESULT_{data_name}/{source}/Final_Label_Tree/{model}/{num}"
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
    if root in collected:
        return collected
    collected.add(root)
    for child in parent_map.get(root, []):
        collect_tree_nodes(child, parent_map, collected)
    return collected

def merge_subtree(i, j, parent_map, label_list: list[data_label], tree_groups):
    node1_parent = [parent for parent, children in parent_map.items() if label_list[i].name in children and parent != ""]
    node1_parent = node1_parent[0] if node1_parent else None
    print(f"{label_list[i].name}的父节点为{node1_parent}！！！！！！！！")
    parent_map[""] = [data for data in parent_map[""] if data != label_list[j].name]
    node2_subtree = []
    for group in tree_groups:
        if label_list[j].name in group:
            node2_subtree = group
            break
    print(f"子节点{label_list[j].name}的子树为{node2_subtree}！！！！！！！！")
    tree_groups = [group for group in tree_groups if label_list[j].name not in group]
    for k in range(len(label_list)):
        if label_list[k].name == node1_parent:
            print(f"{label_list[i].name}的父节点{node1_parent}已经存放子节点{label_list[j].name}!!!!")
            label_list[k].son_label.append(label_list[j].name)
            break
    print(f"{label_list[j].name}节点的子树已删除！！！！！！！！！！")
    if node1_parent:
        parent_map[node1_parent].append(label_list[j].name)
        label_list[j].build_father(node1_parent)
        for group in tree_groups:
            if label_list[i].name in group:
                group.extend(node2_subtree)
                print(f"新的子树为{group}！！！！！！！！！！")
                break
        print(f"{label_list[j].name}节点的家族树已经并入{label_list[i].name}节点的家族树！！！！")
    return parent_map, label_list, tree_groups

def merge_subtree2(i, j, new_father_label,parent_map, label_list: list[data_label], tree_groups):
    parent_map[""] = [data for data in parent_map[""] if data != label_list[i].name or data != label_list[j].name]
    node1_subtree = []
    for group in tree_groups:
        if label_list[i].name in group:
            node1_subtree = group
            break
    print(f"{label_list[i].name}节点的子树为{node1_subtree}！！！！！！！！")
    node2_subtree = []
    for group in tree_groups:
        if label_list[j].name in group:
            node2_subtree = group
            break
    print(f"{label_list[j].name}节点的子树为{node2_subtree}！！！！！！！！")
    tree_groups = [group for group in tree_groups if label_list[j].name not in group]
    print(f"{label_list[j].name}节点的子树已删除！！！！！！！！！！")
    tree_groups = [group for group in tree_groups if label_list[i].name not in group]
    print(f"{label_list[i].name}节点的子树已删除！！！！！！！！！！")
    parent_map[new_father_label].append(label_list[j].name)
    parent_map[new_father_label].append(label_list[i].name)
    label_list[j].build_father(new_father_label)
    label_list[i].build_father(new_father_label)
    for k in range(len(label_list)):
        if label_list[k].name == new_father_label:
            print(f"父节点{new_father_label}已经存放子节点{label_list[i].name}和{label_list[j].name}!!!!")
            label_list[k].son_label.append(label_list[j].name)
            label_list[k].son_label.append(label_list[i].name)
            break
    for group in tree_groups:
        if new_father_label in group:
            group.extend(node2_subtree)
            group.extend(node1_subtree)
            print(f"新的子树为{group}！！！！！！！！！！")
            break
    print(f"{label_list[i].name}节点和{label_list[j].name}节点的家族树已经并入{label_list[i].name}节点的家族树！！！！")
    return parent_map, label_list, tree_groups

def is_cycle(i, j, new_father_label, label_list: list[data_label], tree_groups):
    node1_subtree = []
    for group in tree_groups:
        if label_list[i].name in group:
            node1_subtree = group
            break
    node2_subtree = []
    for group in tree_groups:
        if label_list[j].name in group:
            node2_subtree = group
            break

    if new_father_label in node1_subtree or new_father_label in node2_subtree:
        return "bad"
    else: return "good"
    
def cluster(exist_words: list ,label_list: list[data_label], model: str, num: int, source: str, data_name: str):
    Final_Label_Tree = f"/home/user1/WR/MED_ROLE_TREE/RESULT_{data_name}/{source}/Final_Label_Tree/{model}/{num}"
    parent_map = defaultdict(list)
    node_map = {}
    for item in label_list:
        parent_map[item.father_label].append(item.name)
        node_map[item.name] = item

    tree_groups = []
    visited = set()
    for item in label_list:
        name = item.name
        if name not in visited and item.father_label == "":
            tree_nodes = collect_tree_nodes(name, parent_map)
            visited.update(tree_nodes)
            tree_groups.append(list(tree_nodes))

    embedding_list = []
    name_list = []
    name_to_index = {}
    no_father_list = []
    for item in label_list:
        vector = np.load(item.word_to_vector)
        embedding_list.append(vector)
        name_list.append(item.name)
        name_to_index[item.name] = len(embedding_list) - 1
        if item.father_label == "": no_father_list.append(item.name)
    
    embedding_list = np.array(embedding_list)

    dist_matrix = cdist(embedding_list, embedding_list, "euclidean")

    for group in tree_groups:
        indices = [name_to_index[name] for name in group if name in name_to_index]
        for i in indices:
            for j in indices:
                if i != j:
                    dist_matrix[i, j] = np.inf

    np.fill_diagonal(dist_matrix, np.inf)

    while len(no_father_list) > 1:
        print(f"当前没有父节点的节点数量还有{len(no_father_list)}")
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        print(name_list[i], name_list[j])

        if label_list[i].father_label != "" and label_list[j].father_label != "":
            print("两个节点都有父节点，将两个节点的距离设为inf")
            dist_matrix[i, j] = np.inf
            dist_matrix[j, i] = np.inf
        else:
            if label_list[i].father_label != "":
                print(f"{label_list[i].name}有父节点，下面把{label_list[j].name}并入该节点的父节点！！！")
                parent_map, label_list, tree_groups = merge_subtree(i, j, parent_map, label_list, tree_groups)
                for group in tree_groups:
                    indices = [name_to_index[name] for name in group if name in name_to_index]
                    for z in indices:
                        for x in indices:
                            if z != x:
                                dist_matrix[z, x] = np.inf
                data_logging(label_list, model, num, source, data_name)
                no_father_list = [no_father_list[k] for k in range(len(no_father_list)) if no_father_list[k] != label_list[j].name]
            elif label_list[j].father_label != "":
                print(f"{label_list[j].name}有父节点，下面把{label_list[i].name}并入该节点的父节点！！！")
                parent_map, label_list, tree_groups = merge_subtree(j, i, parent_map, label_list, tree_groups)
                for group in tree_groups:
                    indices = [name_to_index[name] for name in group if name in name_to_index]
                    for z in indices:
                        for x in indices:
                            if z != x:
                                dist_matrix[z, x] = np.inf
                data_logging(label_list, model, num, source, data_name)
                no_father_list = [no_father_list[k] for k in range(len(no_father_list)) if no_father_list[k] != label_list[i].name]
            else:
                n = 0
                print("两个节点都没有父节点，下面进行生成！！！！！")
                fail_label = []
                new_label, n = cluster_word(label_list[i].name, label_list[j].name, model, n)
                print(f"新生成的父节点是{new_label}！！！！！")
                while new_label == None or len(new_label) >= 40:
                    print("新的父节点不合格！！！！！！")
                    new_label, n= cluster_word(label_list[i].name, label_list[j].name, model, n)
                    print(f"新生成的父节点是{new_label}！！！！！")
                combine_label_ui, combine_label_name = check_from_umls(new_label)
                print(combine_label_ui, combine_label_name)
                while combine_label_ui == None and combine_label_name == None or new_label == None or len(new_label) >= 40 or len(combine_label_name) >= 40 or is_exit(exist_words, combine_label_ui) == "bad" and is_cycle(i, j, combine_label_name, label_list, tree_groups) == "bad":
                    print("新的父节点不合格！！！！！！")
                    print(combine_label_ui, combine_label_name)
                    if new_label != None: fail_label.append(new_label)
                    fail_label = list(set(fail_label))
                    new_label, n = cluster_word_plus(label_list[i].name, label_list[j].name, fail_label, model, n)
                    if new_label == None: continue
                    combine_label_ui, combine_label_name = check_from_umls(new_label)
                    print(f"新生成的父节点是{new_label}！！！！！")
                if is_exit(exist_words, combine_label_ui) == "bad":
                    print(f"生成的新节点{combine_label_name}存在于之前的树中且不在这两个节点的树中，现在把两个节点放入该节点的子树！！！")  
                    parent_map, label_list, tree_groups = merge_subtree2(i, j, combine_label_name, parent_map, label_list, tree_groups)
                    for group in tree_groups:
                        indices = [name_to_index[name] for name in group if name in name_to_index]
                        for z in indices:
                            for x in indices:
                                if z != x:
                                    dist_matrix[z, x] = np.inf
                    data_logging(label_list, model, num, source, data_name)
                    no_father_list = [no_father_list[k] for k in range(len(no_father_list)) if no_father_list[k] != label_list[i].name]
                    no_father_list = [no_father_list[k] for k in range(len(no_father_list)) if no_father_list[k] != label_list[j].name]
                elif is_exit(exist_words, combine_label_ui) == "good":
                    print("1")
                    exist_words.append(combine_label_ui)
                    exist_words, label_list = add_new_label(label_list[i].name, label_list[j].name, combine_label_name, combine_label_ui, label_list, exist_words, Final_Label_Tree)
                    parent_map = defaultdict(list)
                    node_map = {}
                    for item in label_list:
                        parent_map[item.father_label].append(item.name)
                        node_map[item.name] = item
                    tree_groups = []
                    visited = set()
                    for item in label_list:
                        name = item.name
                        if name not in visited and item.father_label == "":
                            tree_nodes = collect_tree_nodes(name, parent_map)
                            visited.update(tree_nodes)
                            tree_groups.append(list(tree_nodes))
                    embedding_list = []
                    name_list = []
                    name_to_index = {}
                    no_father_list = []
                    for item in label_list:
                        vector = np.load(item.word_to_vector)
                        embedding_list.append(vector)
                        name_list.append(item.name)
                        name_to_index[item.name] = len(embedding_list) - 1
                        if item.father_label == "": no_father_list.append(item.name)
                    embedding_list = np.array(embedding_list)
                    dist_matrix = cdist(embedding_list, embedding_list, "euclidean")
                    for group in tree_groups:
                        indices = [name_to_index[name] for name in group if name in name_to_index]
                        for z in indices:
                            for x in indices:
                                if z != x:
                                    dist_matrix[z, x] = np.inf
                    np.fill_diagonal(dist_matrix, np.inf)
                    data_logging(label_list, model, num, source, data_name)

    return exist_words, label_list

        


