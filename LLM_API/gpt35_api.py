from openai import OpenAI
import httpx

api_key="",

def chat_with_gpt(prompt):
    response = OpenAI.chat.completions.create(
        temperature=0,
        max_tokens=2000,
        top_p = 1,
        model="gpt-3.5-turbo",  # 选择模型
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content