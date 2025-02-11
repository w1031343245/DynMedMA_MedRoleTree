class data_label:
    ui: str
    name: str
    father_label: str
    son_label: list
    word_to_vector: str

    def __init__(self, ui: str, name: str, father_label: str = "", son_label:list = [], word_to_vector: str="") -> None:
        self.ui = ui
        self.name = name
        self.father_label = father_label
        self.son_label = son_label
        self.word_to_vector = word_to_vector

    def build_father(self, father_label: str):
        self.father_label = father_label

    def build_son(self, son_label: list):
        self.son_label = son_label

    def build_word_to_vector(self, vector_dir:str):
        self.word_to_vector = vector_dir