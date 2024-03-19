# -!- coding: utf-8 -!-
import json
import os
from metric import BatchEvaluation
import argparse
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def batch_evalution(dataset, start_id, end_id, bs_true):
    in_file = os.path.join("../output/ACoT", "xsum_en_fr_output_3.5_3.5.json")
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)["xsum"]

    eva_ori_std = BatchEvaluation()  # (original ref. summary) vs. (GPT-3 std. summary)
    eva_ori_cot = BatchEvaluation()  # (original ref. summary) vs. (GPT-3 cot summary)
    eva_new_std = BatchEvaluation()  # (element-aware ref. summary) vs. (GPT-3 std. summary)
    eva_new_cot = BatchEvaluation()  # (element-aware ref. summary) vs. (GPT-3 cot summary)

    for i in range(start_id, end_id + 1):
        # if i == 4 or i == 27:
        #     continue  # Skip iteration when i is 4

        print(i)
        # ori_ref = " ".join(jieba.lcut(data[i]["original_summary"]))
        ori_ref = " ".join(tokenizer.tokenize(data[i]["original_summary"]))
        # ori_ref = data[i]["original_summary"]
        print(ori_ref )
        # new_ref = " ".join(jieba.lcut(data[i]["element-aware_summary"]))
        new_ref = " ".join(tokenizer.tokenize(data[i]["element-aware_summary"]))
        # new_ref = data[i]["element-aware_summary"]
        print(new_ref)
        # std_pred = " ".join(jieba.lcut(data[i]["gpt3_summary"]))
        std_pred = " ".join(tokenizer.tokenize(data[i]["gpt3_summary"]))
        # std_pred = data[i]["gpt3_summary"]
        print(std_pred)
        # cot_pred = " ".join(jieba.lcut(data[i]["gpt3_cot_summary"]))
        cot_pred = " ".join(tokenizer.tokenize(data[i]["gpt3_cot_summary"]))
        # cot_pred = data[i]["gpt3_cot_summary"]
        print(cot_pred)

        if ori_ref == "" or new_ref == "" or std_pred == "" or cot_pred == "":
            continue

        # eva_ori_std.set_text(ori_ref, std_pred)
        # eva_ori_std.get_rouge_score()
        # if bs_true: eva_ori_std.get_bs_score()
        #
        # eva_ori_cot.set_text(ori_ref, cot_pred)
        # eva_ori_cot.get_rouge_score()
        # if bs_true: eva_ori_cot.get_bs_score()
        #
        eva_new_std.set_text(new_ref, std_pred)
        eva_new_std.get_rouge_score()
        if bs_true: eva_new_std.get_bs_score()

        eva_new_cot.set_text(new_ref, cot_pred)
        eva_new_cot.get_rouge_score()
        if bs_true: eva_new_cot.get_bs_score()

    # print(f"original ref. summary vs. GPT-3 std. summary:\n"
    #       f"batch size:{eva_ori_std.call_time_rs}\n"
    #       f"r1: {eva_ori_std.total_r1/eva_ori_std.call_time_rs}\n"
    #       f"r2: {eva_ori_std.total_r2/eva_ori_std.call_time_rs}\n"
    #       f"rl: {eva_ori_std.total_rl/eva_ori_std.call_time_rs}\n"
    #       f"bs: {eva_ori_std.total_bs/eva_ori_std.call_time_bs}\n")
    #
    # print(f"original ref. summary vs. GPT-3 cot summary:\n"
    #       f"batch size:{eva_ori_cot.call_time_rs}\n"
    #       f"r1: {eva_ori_cot.total_r1 / eva_ori_cot.call_time_rs}\n"
    #       f"r2: {eva_ori_cot.total_r2 / eva_ori_cot.call_time_rs}\n"
    #       f"rl: {eva_ori_cot.total_rl / eva_ori_cot.call_time_rs}\n"
    #       f"bs: {eva_ori_cot.total_bs / eva_ori_cot.call_time_bs}\n")
    #
    print(f"element-aware ref. summary vs. GPT-3 std. summary:\n"
          f"batch size:{eva_new_std.call_time_rs}\n"
          f"r1: {eva_new_std.total_r1 / eva_new_std.call_time_rs}\n"
          f"r2: {eva_new_std.total_r2 / eva_new_std.call_time_rs}\n"
          f"rl: {eva_new_std.total_rl / eva_new_std.call_time_rs}\n"
          f"bs: {eva_new_std.total_bs / eva_new_std.call_time_bs}\n")

    print(f"element-aware ref. summary vs. GPT-3 cot summary:\n"
          f"batch size:{eva_new_cot.call_time_rs}\n"
          f"r1: {eva_new_cot.total_r1 / eva_new_cot.call_time_rs}\n"
          f"r2: {eva_new_cot.total_r2 / eva_new_cot.call_time_rs}\n"
          f"rl: {eva_new_cot.total_rl / eva_new_cot.call_time_rs}\n"
          f"bs: {eva_new_cot.total_bs / eva_new_cot.call_time_bs}\n")


if __name__ == '__main__':
    model_name = "Krystalan/PISCES"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--dataset", type=str, default="cnndm",
                        choices=["cnndm", "xsum"], help="dataset source")
    parser.add_argument("--start_id", type=int, default="0")
    parser.add_argument("--end_id", type=int, default="199")
    parser.add_argument("--bs_true", type=bool, default=False)
    args = parser.parse_args()
    #args.end_id = args.start_id
    batch_evalution(dataset=args.dataset, start_id=args.start_id, end_id=args.end_id, bs_true=args.bs_true)
