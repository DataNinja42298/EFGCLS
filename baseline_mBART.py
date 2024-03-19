from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_korean-large-generic')
result = ner_pipeline('국립진주박물관은 1984년 11월 2일 개관하였으며 한국 전통목조탑을 석조 건물로 형상화한 것으로 건축가 김수근 선생의 대표적 작품이다 .')

print(result)
# {'output': [{'type': 'PER', 'start': 59, 'end': 62, 'span': '김수근'}]}


# import openai
# # -!- coding: utf-8 -!-
# import json
# import os
# import time
# from arguments import parse_arguments
# import re
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
#
# from demo import gpt_35_api_get_final_answer
#
# def mbart_api_get_final_answer(x):
#     try:
#         input_ids = tokenizer(x,
#                               return_tensors="pt",
#                               max_length=512,
#                               padding="max_length",
#                               truncation=True)["input_ids"]
#         output_ids = model.generate(input_ids,
#                                     forced_bos_token_id=tokenizer.lang_code_to_id["ja_XX"],
#                                     max_length=150,
#                                     num_beams=4,
#                                     length_penalty=2.0,
#                                     no_repeat_ngram_size=3)
#         summary = tokenizer.decode(output_ids[0],
#                                    skip_special_tokens=True,
#                                    clean_up_tokenization_spaces=False)
#         return summary
#     except Exception as err:
#         return f'OpenAI API 异常: {err}'
#
# def get_llm_summary(args):
#     in_file = os.path.join("./data", "xsum_en_ja_element_aware.json")
#     with open(in_file, "r", encoding="utf-8") as f:
#         if "xsum" in in_file:
#             data = json.load(f)["xsum"]
#             data_output = {"xsum": []}
#         else:
#             raise "Invalid Dataset!"
#
#     for i in range(0, 200):
#         print("第几条数据:", i)
#         src = data[i]["src"]
#         ori_sum = data[i]["original_summary"]
#         new_sum = data[i]["element-aware_summary"]
#
#         pred_std = mbart_api_get_final_answer(src)
#         print('mBART_ori_sum:', pred_std)
#
#         data_output["xsum"].append({"id": i,
#                                      "src": src,
#                                      "original_summary": " ",
#                                      "element-aware_summary": new_sum,
#                                      "gpt3_summary": " ",
#                                      "gpt3_cot_summary": pred_std})
#
#     data_output = json.dumps(data_output, indent=2)
#     with open("output/mBART/xsum_en_ja_output.json", "w", newline='\n') as g:
#         g.write(data_output)
#
# if __name__ == '__main__':
#     args = parse_arguments()
#
#     model_name = "Krystalan/PISCES"
#     model = MBartForConditionalGeneration.from_pretrained(model_name)
#     tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
#
#     # tokens = tokenizer.tokenize("I like you")
#     # print(tokens)
#
#     get_llm_summary(args)
#
#     print("回答结束！")
