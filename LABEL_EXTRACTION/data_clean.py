import re

def define(word: str):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', word)
    clean_text = re.sub(r'[\*\-\n\'"]', '', clean_text)
    return clean_text

def add_space_to_conmbined_words(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def data_clean(word: str):
    clean_text = define(word)
    word = clean_text
    return word


if __name__ == "__main__":
    print(data_clean("placenta; dysfunction"))