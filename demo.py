import openai
# -!- coding: utf-8 -!-
import json
import os
import time
from arguments import parse_arguments
# from translate_chrome import translate

# openai.api_key = " "

# def gpt_35_api_get_final_answer(x):
#     try:
#         response = openai.Completion.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=x,
#             max_tokens=2048,
#             temperature=0,
#             stop=None
#         )
#         return response["choices"][0]["text"]
#     except Exception as err:
#         return f'OpenAI API 异常: {err}'

# def gpt_35_api_get_final_answer(x):
#     try:
#         response = openai.ChatCompletion.create(
#             # model="gpt-4-1106-preview",
#             # model="gpt-4-0613",
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": x},],
#         )
#         return response["choices"][0].message.content
#     except Exception as err:
#         return f'OpenAI API 异常: {err}'

def gpt_35_api_get_final_answer(x):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            # model="gpt-4",
            messages=[{"role": "user", "content": x},],
        )
        return response["choices"][0].message.content
    except Exception as err:
        return f'OpenAI API 异常: {err}'

# def gpt_35_api_get_final_answer_sam(x, num_samples=2):
#     try:
#         answers = ""
#         for i in range(num_samples):
#             response = openai.Completion.create(
#                 model="text-davinci-002",
#                 prompt=x,
#                 max_tokens=2048,
#                 temperature=0.0 + i * 0.3
#             )
#             answers += response["choices"][0]["text"] + "\n"
#         return answers
#     except Exception as err:
#         return f'OpenAI API 异常: {err}'

def get_llm_summary(args):
    in_file = os.path.join("./data", "cnndm_element_aware.json")
    with open(in_file, "r", encoding="utf-8") as f:
        if "cnndm" in in_file:
            data = json.load(f)["cnndm"]
            data_output = {"cnndm": []}
        else:
            raise "Invalid Dataset!"

    for i in range(args.start_id, args.end_id + 1):
        src = data[i]["src"]
        ori_sum = data[i]["original_summary"]
        new_sum = data[i]["element-aware_summary"]

        # x = "Article: " + src + "\n" + args.std_prompt
        # src_translate = gpt_35_api_get_final_answer(x)
        # print('src_translate:', src_translate)

        # print(args.std_prompt)
        # y = ori_sum + "\n" + args.std_prompt
        # ori_translate = gpt_35_api_get_final_answer(y)
        # print('ori_translate:', ori_translate)

        z = new_sum + "\n" + args.std_prompt
        new_translate = gpt_35_api_get_final_answer(z)
        print('new_translate:', new_translate)

        data_output["cnndm"].append({"id": i,
                                     "src": src,
                                     "original_summary": " ",
                                     "element-aware_summary": new_translate,})

        time.sleep(10)  # 休息5分钟（300秒

    data_output = json.dumps(data_output, indent=2)
    with open("translate/cnndm_en_zh_element_aware.json", "w", encoding="utf-8") as g:
        g.write(data_output)

if __name__ == '__main__':
    args = parse_arguments()
    # get_llm_summary(args)
    in_file = os.path.join("./output", "ACoT/1_cnndm_output_3_3.json")
    with open(in_file, "r", encoding="utf-8") as f:
        if "cnndm" in in_file:
            data = json.load(f)["cnndm"]
            data_output = {"cnndm": []}
        elif "xsum" in in_file:
            data = json.load(f)["xsum"]
            data_output = {"xsum": []}
        else:
            raise "Invalid Dataset!"

    for i in range(0, 1):
        print("第几条数据:", i)
        src = data[i]["src"]
        ori_sum = data[i]["original_summary"]
        new_sum = data[i]["element-aware_summary"]

        gpt__sum = data[i]["gpt3_cot_summary"]

        f = gpt__sum + "\n" + args.cot_fina
        print("模型输入:", f)
        pred_fina = gpt_35_api_get_final_answer(f)
        print("summary:", pred_fina)
