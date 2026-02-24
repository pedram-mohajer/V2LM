from openai import OpenAI
from io import BytesIO
import base64

client = OpenAI(
    base_url="http://localhost:25111",  # change if needed
    api_key="fake-key",
)

def file_to_base64_binary(file_path: str):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def main(model: str = " Efficient-Large-Model/NVILA-15B", stream: bool = True):
    prompt = "In the video, a person is making coffee. What specific action are they performing during the coffee-making process?"

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{file_to_base64_binary('./output1.mp4')}",

                        },
                        "frames": 16, 
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        model=model,
        stream=stream,
    )

    if stream:
        for chunk in response:
            print(chunk.choices[0].delta.content, end="")
        print()
    else:
        print(response.choices[0].message.content[0]["text"])

if __name__ == "__main__":
    import fire
    fire.Fire(main)
