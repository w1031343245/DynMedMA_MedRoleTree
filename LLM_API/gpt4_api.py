from openai import OpenAI
import httpx


client = OpenAI(
    base_url="https://api.xty.app/v1", 
    api_key="sk-ePi7oFmEBvWsJeTT6534E6Fe707640C980190933752a77F1",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

def chat_with_gpt4(prompt):
    response = client.chat.completions.create(
        temperature=0,
        max_tokens=100,
        top_p = 1,
        model="gpt-4",  # 选择模型
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content


