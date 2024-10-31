from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.2-3b-instruct",
  messages=[{"role":"user","content":"what is machine learning and AI"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

