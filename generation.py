# -!- coding: utf-8 -!-
import json
import os
import time
from api_request import Decoder
from arguments import parse_arguments
from demo import gpt_35_api_get_final_answer
# from translate_chrome import translate

def get_llm_summary(args):
    in_file = os.path.join("./data", "xsum_en_fr_element_aware.json")
    with open(in_file, "r", encoding="utf-8") as f:
        if "cnndm" in in_file:
            data = json.load(f)["cnndm"]
            data_output = {"cnndm": []}
        elif "xsum" in in_file:
            data = json.load(f)["xsum"]
            data_output = {"xsum": []}
        else:
            raise "Invalid Dataset!"

    for i in range(args.start_id, args.end_id + 1):
        print("第几条数据:", i)
        src = data[i]["src"]
        ori_sum = data[i]["original_summary"]
        new_sum = data[i]["element-aware_summary"]

        # x = "Article: " + src + "\n" + args.std_prompt
        # pred_std = gpt_35_api_get_final_answer(x)
        # print("pred_std:", pred_std)
        # time.sleep(10)  # 休息5分钟（300秒
        #
        # # cot方法
        # x = "Article: " + src + "\n" + args.cot
        # ele = gpt_35_api_get_final_answer(x)
        # x = x + ele + "\n" + args.cot_prompt
        # print("提示：", x)
        # time.sleep(10)  # 休息5分钟（300秒
        # pred_cot = gpt_35_api_get_final_answer(x)
        # print("pred_cot：", pred_cot)
        # time.sleep(10)  # 休息5分钟（300秒

        # # 先翻译后摘要
        # x = src + "\n" + args.cot
        # translate = gpt_35_api_get_final_answer(x)
        # print("translate:", translate)
        # x = "Article: " + translate + "\n" + args.cot_prompt
        # pred_cot = gpt_35_api_get_final_answer(x)
        # print("pred_cot:", pred_cot)
        # time.sleep(10)  # 休息5分钟（300秒

        # # 先摘要后翻译
        # x = src + "\n" + args.cot
        # summary = gpt_35_api_get_final_answer(x)
        # print("summary:", summary)
        # time.sleep(10)  # 休息5分钟（300秒
        # x = summary + "\n" + args.cot_prompt
        # pred_cot = gpt_35_api_get_final_answer(x)
        # print("translate:", pred_cot)
        # time.sleep(10)  # 休息5分钟（300秒

        # scot方法
        x = "Article: " + src + "\n" + args.cot
        ele = gpt_35_api_get_final_answer(x)
        print("元素关系:", ele)
        time.sleep(10)  # 休息5分钟（300秒

        y = ele + "\n" + args.cot_sam
        cls = gpt_35_api_get_final_answer(y)
        print("排序前四:", cls)
        time.sleep(10)  # 休息5分钟（300秒

        z = "Article: " + src + "\n" + "Important information in the article is as follows:" + "\n" + cls + "\n" + args.cot_prompt
        print("模型输入:", z)
        pred_cot = gpt_35_api_get_final_answer(z)
        print("summary:", pred_cot)

        f = pred_cot + "\n" + args.cot_fina
        print("模型输入:", f)
        pred_fina = gpt_35_api_get_final_answer(f)
        print("summary:", pred_fina)

        data_output["xsum"].append({"id": i,
                                     "src": src,
                                     "original_summary": ori_sum,
                                     "element-aware_summary": new_sum,
                                     "gpt3_summary": " ",
                                     "gpt3_cot_summary": pred_fina})

        time.sleep(10)  # 休息5分钟（300秒

    print("回答结束！")
    data_output = json.dumps(data_output, indent=2)
    with open("output/ACoT/xsum_en_fr_output_4_4.json", "w", encoding="utf-8") as g:
        g.write(data_output)


if __name__ == '__main__':
    args = parse_arguments()

    get_llm_summary(args)

