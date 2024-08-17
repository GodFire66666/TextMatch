import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import openai

# 传入参数
parser = argparse.ArgumentParser(description="Generate images from text prompts")
parser.add_argument("--data_type", type=str)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--begin_idx", type=int, default=0)

args = parser.parse_args()
data_type = args.data_type
device = f"cuda:{args.device}"
begin_idx = args.begin_idx
print(f"Data type: {data_type}")
print(f"Device: {device}")
print(f"Begin index: {begin_idx}")


api_key = 'sk-xx'
api_base = 'https://xx'
os.environ['OPENAI_API_KEY'] = api_key
os.environ['OPENAI_API_BASE'] = api_base
openai.api_key = api_key
openai.base_url = api_base
client = openai.OpenAI(api_key=api_key, base_url=api_base)


import random
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img

def get_image_from_path(file_path: str):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img

import base64


def encode_image_from_path(image_path):
    """
    对图片文件进行 Base64 编码

    输入：
         - image_path：图片的文件路径
    输出：
         - 编码后的 Base64 字符串
    """
    # 二进制读取模式打开图片文件，
    with open(image_path, "rb") as image_file:
        # 将编码后的字节串解码为 UTF-8 字符串，以便于在文本环境中使用。
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_from_PIL_image(image):
    # 创建一个内存缓冲区
    buffered = BytesIO()
    # 将 PIL 图像对象保存到内存缓冲区中，格式为 JPEG，你也可以选择其他格式
    image.save(buffered, format="JPEG")
    # 获取缓冲区中的字节数据并将其编码为 base64 字符串
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def openai_completion(prompt, engine="gpt-4o", max_tokens=700, temperature=0):    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n", "<|endoftext|>"]
        )
    
    return resp.choices[0].message.content


from visual_scoring.score import (UnifiedQAModel, VQAModel, VS_score_single,
                                  filter_question_and_answers,
                                  get_question_and_answers)

unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200", device=device)
vqa_model = VQAModel("mplug-large", device=device)

def get_VS_result(text, img_path, filtered_questions=None):
    if not filtered_questions:
        # Generate questions with GPT
        gpt_questions = get_question_and_answers(text)

        # Filter questions with UnifiedQA
        filtered_questions = filter_question_and_answers(unifiedqa_model, gpt_questions)
 
        # See the questions
        # print(filtered_questions)

        # calucluate VS score
        result = VS_score_single(vqa_model, filtered_questions, img_path)
        return filtered_questions, result
    else:
        # calucluate VS score
        result = VS_score_single(vqa_model, filtered_questions, img_path)
        return result



def generate_image(prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )

    image_url = response.data[0].url
    img = get_image_from_url(image_url)
    return img

def format_prompt_to_message(user_prompt, previous_prompts, generated_image, num_solutions, result):
    image = encode_image_from_PIL_image(generated_image)

    VS_results = []
    for i, (key, value) in enumerate(result['question_details'].items()):
        VS_result = "Element " + str(i) + "\n"
        VS_result += "Question: " + key + "\n"
        VS_result += "Ground Truth: " + value['answer'] + "\n"
        VS_result += "In the image generated from above prompt, the VQA model identified infer that the answer to the question is: " + value['free_form_vqa'] + "\n"

        VS_results.append(VS_result)

    VS_results = "\n".join(VS_results)

    prompt = f"""
You are an expert prompt optimizer for text-to-image models. Text-to-image models take a text prompt as input and generate images depicting the prompt as output. You are responsible for transforming human-written prompts into improved prompts for text-to-image models. Your responses should be concise and effective.

Your task is to optimize the human initial prompt: "{user_prompt}". Below are some previous prompts along with a breakdown of their visual elements. Each element is paired with a score indicating its presence in the generated image. A score of 1 indicates visual elements matching the human initial prompt, while a score of 0 indicates no match.

Here is the image that the text-to-image model generated based on the initial prompt:
{{image_placeholder}}

Here are the previous prompts and their visual element scores:
## Previous Prompts
{previous_prompts}
## Visual Element Scores
{VS_results}

Generate {num_solutions} paraphrases of the initial prompt which retain the semantic meaning and have higher scores than all the previous prompts. Prioritize optimizing for objects with the lowest scores. Prefer substitutions and reorderings over additions. Please respond with each new prompt in between <PROMPT> and </PROMPT>, for example:
1. <PROMPT>paraphrase 1</PROMPT>
2. <PROMPT>paraphrase 2</PROMPT>
...
{num_solutions}. <PROMPT>paraphrase {num_solutions}</PROMPT>
"""
    text_prompts = prompt.split("{image_placeholder}")

    user_content = [{"type": "text", "text": text_prompts[0]}]
    base64_images = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
                "detail": "high",
            },
        }
    ]
    user_content.extend(base64_images)
    user_content.append({"type": "text", "text": text_prompts[1]})
    messages_template = [{"role": "user", "content": user_content}]

    return messages_template

def generate_image_chat_response(messages_template, client):
    payload = {
        "model": "gpt-4o",
        "messages": messages_template,
        "max_tokens": 1600,
        "temperature": 0,
        "seed": 2024,
    }
    
    # 调用 OpenAI API 生成回复
    response = client.chat.completions.create(**payload)
    
    # 返回生成的结果
    return response.choices[0].message.content

import re


def extract_prompts(text):
    pattern = r'<PROMPT>(.*?)</PROMPT>'
    prompts = re.findall(pattern, text)
    return prompts



max_retries = 5  # 最大重试次数


def DALLE3_VS(prompt):
    success = False
    retries = 0
    print(f"Generating image for prompt: {prompt}")
    while not success and retries < max_retries:
        try:
            image = generate_image(prompt=prompt)
            success = True
            print("Image generated successfully!")
        except Exception as e:
            retries += 1
            print(f"Error: {e}")
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                # time.sleep(1)  # 等待 1 秒后重试
            else:
                print("Max retries reached. Exiting.")
                break
    if not success:
        print("Failed to generate image. Exiting.")
        return

    success = False
    retries = 0
    print("Calculating VS score...")
    while not success and retries < max_retries:
        try:
            filtered_questions, VS_result = get_VS_result(prompt, image)
            success = True
            print(f"\nVS score: {VS_result['VS_score']}")  
        except Exception as e:
            retries += 1
            print(f"Error: {e}")
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                # time.sleep(1)  # 等待 1 秒后重试
            else:
                print("Max retries reached. Exiting.")
                break
    if not success:
        print("Failed to calculate VS score. Exiting.")
        return image


    success = False
    retries = 0
    print("Generating new prompt...")
    while not success and retries < max_retries:
        try:
            formatted_prompt = format_prompt_to_message(user_prompt=prompt, 
                                                        previous_prompts=prompt, 
                                                        generated_image=image, 
                                                        num_solutions=3, 
                                                        result=VS_result)
            generate_prompts = generate_image_chat_response(formatted_prompt, client)
            new_regional_prompt = extract_prompts(generate_prompts)[0]
            success = True
            print("Prompt formatted successfully!")
        except Exception as e:
            retries += 1
            print(f"Error: {e}")
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                # time.sleep(1)  # 等待 1 秒后重试
            else:
                print("Max retries reached. Exiting.")
                break 
    if not success:
        print("Failed to generate new prompt. Exiting.")
        return image

    print(f"New prompt generated: {new_regional_prompt}")    
    try:
        new_image = generate_image(
            prompt=prompt,
        )
    except Exception as e:
        print(f"Error: {e}")
        return image

    new_VS_result = get_VS_result(prompt, new_image, filtered_questions)
    print(f"\nVS score: {new_VS_result['VS_score']}")  

    if new_VS_result['VS_score'] > VS_result['VS_score']:
        return new_image
    else:
        return image

save_dir = f"images/{data_type}/samples"
os.makedirs(save_dir, exist_ok=True)
descriptions = []
with open(f"examples/dataset/{data_type}_val.txt", "r") as f:
    for line in f:
        descriptions.append(line.strip())


from tqdm import tqdm

for idx, prompt in tqdm(enumerate(descriptions[begin_idx:], start=begin_idx)):
    image = DALLE3_VS(prompt)
    filename = f"{prompt}_{idx:06d}.png"
    image.save(os.path.join(save_dir, filename))
    print(f"Image {idx} saved successfully!")




