import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ.get("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"))

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Responde solo con: Conexión exitosa"}
    ]
)

print(message.content[0].text)