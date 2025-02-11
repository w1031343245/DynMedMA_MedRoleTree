from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:8000/v1/',
    api_key='ollama',  # required but ignored
)

def chat_with_qwen(prompt, temperature, max_tokens, top_p):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': "user",
                'content': prompt,
            }
        ],
        model='qwen2.5:7b',
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    return chat_completion.choices[0].message.content
