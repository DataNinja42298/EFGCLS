import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
# -!- coding: utf-8 -!-
import json
import os
import time
from arguments import parse_arguments

# 设置 OpenAI API 密钥和基础 API 地址
# openai.api_key = "sk-oMvS4bIlYOvgaTq00K0A75IGRuh8tvrCFosAKakae34HwIUM"
# openai.api_key = "sk-fR40qbXzdT8s7EUY5uRBJfgDwv6Sh4BMCjEjHTl14h85XarZ"
openai.api_key = "sk-FiRGk4dIm8KqHYubWlJRKWrWKngBL74RtqcGF7polzZnqoMW"
# openai.api_key = "sk-cx9WlHt5UbDTqXXlSCwsSMDuBMokMirwInJ4nljM5WsbqFNh"
openai.api_base = "https://api.chatanywhere.com.cn/v1"

def gpt_35_api_get_final_answer(x):
    try:
        response = openai.ChatCompletion.create(
            # model="gpt-4-1106-preview",
            # model="gpt-4-0613",
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": x},],
        )
        return response["choices"][0].message.content
    except Exception as err:
        return f'OpenAI API 异常: {err}'

def mT5_api_get_final_answer(x):
    try:
        WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

        get_lang_id = lambda lang: tokenizer._convert_token_to_id(
            model.config.task_specific_params["langid_map"][lang][1]
        )

        target_lang = "japanese"  # for a list of available language names see below

        input_ids = tokenizer(
            [WHITESPACE_HANDLER(x)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            decoder_start_token_id=get_lang_id(target_lang),
            max_length=84,
            no_repeat_ngram_size=2,
            num_beams=4,
        )[0]

        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return summary
    except Exception as err:
        return f'OpenAI API 异常: {err}'

def get_llm_summary(args):
    in_file = os.path.join("./data", "xsum_en_ja_element_aware.json")
    with open(in_file, "r", encoding="utf-8") as f:
        if "xsum" in in_file:
            data = json.load(f)["xsum"]
            data_output = {"xsum": []}
        else:
            raise "Invalid Dataset!"

    for i in range(0, 200):
        print("第几条数据:", i)
        src = data[i]["src"]
        ori_sum = data[i]["original_summary"]
        new_sum = data[i]["element-aware_summary"]

        pred_std = mT5_api_get_final_answer(src)
        print('mT5_ori_sum:', pred_std)
        # 指定标识符
        identifier = '<extra_id_62>'
        # 使用字符串切片提取标识符之后的字符
        pred_cot = pred_std.split(identifier, 1)[-1]
        print('pred_cot:', pred_cot)

        data_output["xsum"].append({"id": i,
                                     "src": src,
                                     "original_summary": " ",
                                     "element-aware_summary": new_sum,
                                     "gpt3_summary": " ",
                                     "gpt3_cot_summary": pred_cot})

    data_output = json.dumps(data_output, indent=2)
    with open("output/mT5/xsum_en_ja_output.json", "w", newline='\n') as g:
        g.write(data_output)

if __name__ == '__main__':
    args = parse_arguments()

    model_name = "csebuetnlp/mT5_m2m_crossSum_enhanced"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    get_llm_summary(args)

    print("回答结束！")
