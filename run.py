import argparse
from LABEL_EXTRACTION.label_extraction import *
from LABEL_DEFINE.Label import *
from CLUSTER_WITH_LMM.build_tree import *
from MULTI_AGENT.dynatic_domain import *
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="角色树构建所需要的一系列设置")
    parser.add_argument("--model", type=str, default='qwen2.5',help="模型名称")
    parser.add_argument("--num", type=int, default = 300,help="抽取数据集中数据的数量")
    parser.add_argument("--source", type=str, default="train", help="标签来源test/train/book")
    parser.add_argument("--dataset", type=str, default="MedMCQA", help="数据集")

    args = parser.parse_args()
    exist_words = []

    label_list: list[data_label] = []
    file_path = Path(f"/home/user1/WR/MED_ROLE_TREE/RESULT_{args.dataset}/{args.source}/Label_Family_Tree/{args.model}/{args.num}/label_tree_data.jsonl")
    if file_path.exists():
        data = []
        with open(f"/home/user1/WR/MED_ROLE_TREE/RESULT_{args.dataset}/{args.source}/Label_Family_Tree/{args.model}/{args.num}/label_tree_data.jsonl", 'r', encoding='utf-8') as f:
            for item in f:
                data.append(json.loads(item.strip()))
        for da in data:
            lab = data_label(da['ui'], da['name'], da['father_label'], da['son_label'], da['word_to_vector'])
            label_list.append(lab)
            exist_words.append(lab.ui)
        finish_question_num = []
        with open(f"/home/user1/WR/MED_ROLE_TREE/RESULT_{args.dataset}/{args.source}/Label_Family_Tree/{args.model}/{args.num}/sum_list.csv", 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                finish_question_num.append(int(row[0]))
        if len(finish_question_num) < args.num:
            print(f"之前程序发生过中断，现在从上次断开位置开始进行，{len(finish_question_num)}")
            exist_words, label_list  = KE(args.num, args.model, exist_words, label_list, len(finish_question_num), finish_question_num, args.source, args.dataset)
    else:
        exist_words, label_list  = KE(args.num, args.model, exist_words, label_list, 0, [], args.source, args.dataset)
        
    final_label_tree_data_dir = f"/home/user1/WR/MED_ROLE_TREE/RESULT_{args.dataset}/{args.source}/Final_Label_Tree/{args.model}/{args.num}"
    if not os.path.exists(final_label_tree_data_dir): os.makedirs(final_label_tree_data_dir, exist_ok=True)
    file2_path = Path(f"{final_label_tree_data_dir}/label_tree_data.jsonl")
    if file2_path.exists():
        print("树生成操作出现过中断，现在继续！！！！")
        data = []
        label_list =[]
        exist_words = []
        with open(f"/home/user1/WR/MED_ROLE_TREE/RESULT_{args.dataset}/{args.source}/Final_Label_Tree/{args.model}/{args.num}/label_tree_data.jsonl", 'r', encoding='utf-8') as f:
            for item in f:
                data.append(json.loads(item.strip()))
        for da in data:
            lab = data_label(da['ui'], da['name'], da['father_label'], da['son_label'], da['word_to_vector'])
            label_list.append(lab)
            exist_words.append(lab.ui)
    exist_words, label_list = cluster(exist_words, label_list, args.model, args.num, args.source, args.dataset)

    
if __name__ == "__main__":
    main()