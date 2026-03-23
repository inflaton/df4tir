import base64
import argparse
from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Image captioning using OpenAI API")
    parser.add_argument("--image_url", help="URL of the image to caption")
    parser.add_argument("--api-key", default="ollama", help="OpenAI API key")
    parser.add_argument(
        "--api-base", default="http://localhost:8000/v1", help="OpenAI API base URL"
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    detailed_prompt = """Act as an OCR assistant. Analyze the provided image and:
    1. Recognize all visible text in the image as accurately as possible.
    2. Maintain the original structure and formatting of the text.
    3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
    Provide only the transcription without any additional comments."""

    try:
        chat_response = client.chat.completions.create(
            # model="HuggingFaceTB/SmolVLM-Instruct",
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            temperature=0.4,
            top_p=0.8,
            frequency_penalty=1.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detailed_prompt},
                        {"type": "image_url", "image_url": {"url": args.image_url}},
                    ],
                }
            ],
        )
        print("OCR Result:", chat_response.choices[0].message.content)
    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")


if __name__ == "__main__":
    main()
