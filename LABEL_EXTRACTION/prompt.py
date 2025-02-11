def KNOWLEDGE_EXTRACTION(question: str):
    question_domain_format = "Knowledge labels: label"
    ROLE_PROMPT = "You are a knowledgeable person who can accurately identify the most important medical knowledge labels in the question.\n"
    PROMPT_TEMPLATE = f"You need to complete the following steps:\n"\
        f"1. Please read the question carefully. The question is: '''{question}'''.\n"\
        f"2. Please identify a key medical knowledge labels that are each a medical term, and do not exceed 15 characters.\n"\
        f"3. Each label must be an independent noun, and should not contain other labels or phrases. Avoid using multi-word phrases.\n"\
        f"4. Output the knowledge labels in the following format: {question_domain_format}.\n"
    
    return ROLE_PROMPT + "\n\n" + PROMPT_TEMPLATE

def KNOWLEDGE_EXTRACTION_GPT(question: str):
    instruction = f"""
    Instruction:
    You are an expert tasked with extracting key medical knowledge labels from the given question.

    Requirements:
    1. Read the given question carefully:
       Question: '''{question}'''

    2. Extract key medical knowledge labels from the question, adhering to the following rules:
       - Each label must be an independent noun (a single-word term).
       - Avoid multi-word phrases or combined terms.
       - Each label must not exceed 15 tokens.

    3. Output the labels in the following format:
       ```
       Knowledge labels: [label1, label2, ...]
       ```

    Example:
    Input Question:
    ```
    The patient exhibits fever, cough, and difficulty breathing. The doctor suspects it might be pneumonia.
    ```

    Expected Output:
    ```
    Knowledge labels: fever
    ```
    """
    return instruction

def KNOWLEDGE_EXTRACTION_GPT_REPEAT(question: str, used_labels: list):
    used_labels_str = ", ".join(used_labels) if used_labels else "None"

    instruction = f"""
    Instruction:
    You are tasked with extracting new key medical knowledge labels from a question, ensuring that 
    previously identified labels are not repeated.

    Requirements:
    1. Read the given question carefully:
       Question: '''{question}'''

    2. Identify new key medical knowledge labels:
       - Each label must be a single-word medical term (independent noun).
       - Avoid multi-word phrases or combined terms.
       - Each label must not exceed 15 tokens.
       - Do not include any of the following labels that have already been identified: 
         {used_labels_str}

    3. Output the new knowledge labels in the following format:
       ```
       Knowledge labels: [label1, label2, ...]
       ```
       - Separate the labels with commas.
       - Do not include any additional text or explanations.

    Example:

    Input Question:
    ```
    The patient is experiencing nausea, fatigue, and abdominal pain, which could indicate gastritis.
    ```
    Previously Identified Labels:
    ```
    nausea, fatigue
    ```

    Expected Output:
    ```
    Knowledge labels: pain
    ```
    """
    return instruction

def DOMAIN_EXTRACTION_GPT(question: str):
    instruction = f"""
    Instruction:
    You are an expert tasked with extracting key medical domain terms from the given question.

    Requirements:
    1. Read the given question carefully:
       Question: '''{question}'''

    2. Extract key medical domain terms from the question, adhering to the following rules:
       - Each term must be an independent noun (a single-word term).
       - Avoid multi-word phrases or combined terms.
       - Each term must not exceed 30 tokens.

    3. Output the terms in the following format:

    Medical domain terms: [domain1, domain2, ...]

    Example:
    Input Question:
    
    The patient exhibits fever, cough, and difficulty breathing. The doctor suspects it might be pneumonia.
    
    Expected Output:
    
    Medical domain terms: fever
    """
    return instruction


def DOMAIN_EXTRACTION_GPT_REPEAT(question: str, used_labels: list):
    used_labels_str = ", ".join(used_labels) if used_labels else "None"

    instruction = f"""
    Instruction:
    You are tasked with extracting new key medical domain terms from a question, ensuring that 
    previously identified terms are not repeated.

    Requirements:
    1. Read the given question carefully:
       Question: '''{question}'''

    2. Identify new key medical domain terms:
       - Each term must be a single-word medical term (independent noun).
       - Avoid multi-word phrases or combined terms.
       - Each term must not exceed 15 tokens.
       - Do not include any of the following terms that have already been identified: 
         {used_labels_str}

    3. Output the new terms in the following format:
       ```
       Medical domain terms: [term1, term2, ...]
       ```
       - Separate the terms with commas.
       - Do not include any additional text or explanations.

    Example:

    Input Question:
    ```
    The patient is experiencing nausea, fatigue, and abdominal pain, which could indicate gastritis.
    ```
    Previously Identified Terms:
    ```
    nausea, fatigue
    ```

    Expected Output:
    ```
    Medical domain terms: pain
    ```
    """
    return instruction
