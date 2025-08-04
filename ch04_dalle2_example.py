from openai import OpenAI

client = OpenAI(api_key="--------")

response = client.images.generate(
    model="dall-e-2",
    prompt="A futuristic city at night",
    n=1,
    size="1024x1024"
)

print("생성된 이미지 URL:", response.data[0].url)
