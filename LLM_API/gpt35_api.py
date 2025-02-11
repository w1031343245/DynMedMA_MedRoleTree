from openai import OpenAI
import httpx


client = OpenAI(
    base_url="https://api.xty.app/v1", 
    api_key="sk-dfzX2spL5kaSkoe6760310B5B77f4533B367BaDe9c436573",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        temperature=0,
        max_tokens=2000,
        top_p = 1,
        model="gpt-3.5-turbo",  # 选择模型
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content


def chat_with_gpt2(role, prompt):
    response = client.chat.completions.create(
        temperature=0,
        max_tokens=2000,
        top_p = 1,
        model="gpt-3.5-turbo",  # 选择模型
        messages=[
            {"role": role, "content": prompt},
        ]
    )

    return response.choices[0].message.content