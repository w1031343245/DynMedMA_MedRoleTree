def LLM_CLUSTER(label1: str, label2: str):
    output_format = """Label: label1, label2
cluster_label: label3"""
    ROLE_PROMPT = "Please act as a 'Label Hierarchizer' to merge two given labels into a higher-level label, ensuring that the new label is highly related in theme."
    PROMPT_TEMPLATE = f"""Please follow these rules:
1. The new label must accurately represent the common theme of both '{label1}' and '{label2}'.
2. The label should be concise, relevant, and use standardized terminology that is widely accepted.
3. Ensure that the new label can be found on Wikipedia and is a recognized term in its respective domain.
4. The format for your output should be as follows:
Output format:
        {output_format}"""
    
    return ROLE_PROMPT + '\n\n' + PROMPT_TEMPLATE

def LLM_CLUSTER_plus(label1: str, label2: str, fail_label: list):
    fail_label_str = ", ".join(sorted(set(fail_label)))

    ROLE_PROMPT = "You are a 'Label Hierarchizer'. Your task is to merge two given labels into a concise, higher-level label that accurately represents their common theme."

    PROMPT_TEMPLATE = f"""Please follow these rules:
1. Identify a single, higher-level label that accurately represents the common theme of '{label1}' and '{label2}'.
2. Do not use the following labels as the new label: {fail_label_str}.
3. The new label must:
   - Be a single concept (not multiple terms joined by conjunctions like 'and').
   - Be concise, relevant, and widely accepted in its domain.
   - Exist as a recognized term (e.g., verifiable on Wikipedia or similar authoritative sources).
4. Provide only the final label as the answer.
5. Attention: Do not use the generated tags above as higher-level tags.
Output format:
Label: label1, label2
cluster_label: label3"""

    # Return the full prompt
    return ROLE_PROMPT + '\n\n' + PROMPT_TEMPLATE

def LLM_CLUSTER_GPT(label1: str, label2: str):
    instruction = f"""
   Instruction:
   You are an expert tasked with merging two given labels into a single, higher-level label that accurately represents their common theme.

   Requirements:
   1. Read the following labels carefully:
      - Label 1: '''{label1}'''
      - Label 2: '''{label2}'''

   2. Identify a single, higher-level label that captures the common theme between the two labels. This new label must adhere to the following rules:
      - The label should be 'concise', 'highly relevant', and 'widely accepted' in its domain.
      - If the labels seem too specific or niche, increase the abstraction level to a broader concept. For example, if the labels are very detailed, consider using a more general term that encompasses both. Aim to avoid overly specific labels unless absolutely necessary.
      
   3. When considering abstraction, remember:
      - The new label should be a 'single term' that groups both labels under a common umbrella. 
      - 'Avoid using overly niche terms'. For instance, if combining "Diabetes" and "Hypertension", don’t use terms like "Chronic Kidney Disease". Instead, aim for a broader concept like "Cardiometabolic Diseases."

   4. The cluster label must not exceed 30 characters in length.

   5. Output the final label only. Do 'not' provide any additional explanations or rationale.

   6. Output format:
       ```
       Label: label1, label2
       Cluster label: new_label
       ```

    Example:
    Input:
    ```
    Label 1: Hypertension
    Label 2: Heart Disease
    ```
    Expected Output:
    ```
    Label: Hypertension, Heart Disease
    Cluster label: Cardiovascular Disease
    ```

    """
    return instruction



def LLM_CLUSTER_plus_GPT(label1: str, label2: str, fail_label: list):
    fail_label_str = ", ".join(sorted(set(fail_label)))

    instruction = f"""
   Instruction:
You are tasked with merging two given labels into a single, higher-level concept that reflects their common theme.

Steps:
1. Review the following labels:
   Label 1: '''{label1}'''
   Label 2: '''{label2}'''

2. Create a higher-level label that captures the shared theme of the two labels. The new label must:
   Exclude the following labels: '''{fail_label_str}'''. These labels should not be considered, even if they seem related.

3. Guidelines for abstraction:
   Choose a 'general concept' that groups both labels under a broader umbrella. For instance, "Musculoskeletal Disorders" is a better fit for "Arthritis" and "Osteoporosis" than something overly specific like "Bone Degeneration".

4. The final label must be concise, with no more than 30 characters.

5. Only output the final label. Do not provide additional explanations or reasoning.

6. Important: Avoid using the labels provided '''{fail_label}''' above as part of the new, higher-level label.

Output Format:

   7. Output format:
       ```
       Label: label1, label2
       Cluster label: new_label
       ```
    """
    return instruction


