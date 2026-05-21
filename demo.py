from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

completion = client.chat.completions.create(
    # model="deepseek-ai/deepseek-v4-pro",
    # model="z-ai/glm-5.1",
    model="deepseek-ai/deepseek-v4-flash",
    # model="qwen/qwen3-coder-480b-a35b-instruct",
    # model="minimaxai/minimax-m2.7",
    # model="google/gemma-4-31b-it",
    messages=[{"role": "user", "content": "你是谁"}],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    stream=True
)

for chunk in completion:
    if not getattr(chunk, "choices", None):
        continue
    reasoning = getattr(chunk.choices[0].delta, "reasoning", None) or getattr(chunk.choices[0].delta,
                                                                              "reasoning_content", None)
    if reasoning:
        print(reasoning, end="")
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


