import argparse
import base64
import os
import magic
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
cwd = os.getcwd()

from langchain_community.llms import VLLM


def ocr_tool(model: any, image_path: str) -> str:
    """OCR tools parse invoice images to produce useful data for reconciliation"""

    file_path = f"{cwd}/dataset/attachments/{image_path}"

    ocr_prompt = """Act as an OCR assistant. Analyze the provided image and:
    1. Recognize all visible text in the image as accurately as possible.
    2. Maintain the original structure and formatting of the text.
    3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
    Provide only the transcription without any additional comments."""

    print(
        "Tool",
        f"""
    Invoke OCR for {image_path}
    """,
    )

    encoded_img_string = ""
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)

    # router = ModelRouter()
    # llm_model = router.get_model(
    #     model_type=os.environ["VISION_MODEL"], temperature=0, is_vision=True
    # )

    with open(file_path, "rb") as image_file:
        # model = llm_model["model"]
        encoded_img_string = base64.b64encode(image_file.read()).decode("utf-8")
        print("len:", len(encoded_img_string))

    def invoker(data):
        image = data["image"]
        mime_type = data["mime_type"]
        prompt = data["prompt"]

        img_data_str = f"data:{mime_type};base64,{image}"

        return [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": (
                            # {"url": img_data_str}
                            # if llm_model["provider"] == "openai"
                            # else
                            img_data_str
                        ),
                    },
                    {"type": "text", "text": prompt},
                ]
            )
        ]

    chain = invoker | model | StrOutputParser()
    result = chain.invoke(
        {"image": encoded_img_string, "mime_type": mime_type, "prompt": ocr_prompt}
    )

    print(
        "Tool",
        f"""
        OCR result: {result}
        """,
    )

    return {"content": result}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Image captioning using OpenAI API")
    parser.add_argument("--image_path", help="URL of the image to caption")
    args = parser.parse_args()

    llm = VLLM(
        # model="HuggingFaceTB/SmolVLM-Instruct",
        # model="Qwen/Qwen2.5-VL-3B-Instruct",
        # model="Qwen/Qwen2.5-VL-32B-Instruct",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,  # mandatory for hf models
        max_seq_len=32768,
        max_new_tokens=1024,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        # vllm_kwargs={"max_model_len": 32768}
    )

    result = ocr_tool(llm, args.image_path)
    print(result)


if __name__ == "__main__":
    main()
