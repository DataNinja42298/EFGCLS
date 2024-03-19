# -!- coding: utf-8 -!-
import argparse


def get_prompt():
    std_generation_cnndm_prompt = open("./prompts/std_generation_cnndm.txt", encoding="utf-8").read()
    std_generation_xsum_prompt = open("./prompts/std_generation_xsum.txt", encoding="utf-8").read()
    cot_generation_cnndm_prompt = open("./prompts/cot_generation_cnndm.txt", encoding="utf-8").read()
    cot_generation_xsum_prompt = open("./prompts/cot_generation_xsum.txt", encoding="utf-8").read()
    cot_sample_prompt = ""
    cot_extraction_prompt = ""
    cot_check_prompt = ""
    cot_fina_prompt = ""
    for line in open("./prompts/cot_element_extraction.txt", encoding="utf-8"):
        cot_extraction_prompt += line
    for line_1 in open("./prompts/cot_sample_cnndm.txt", encoding="utf-8"):
        cot_sample_prompt += line_1
    for line_2 in open("./prompts/cot_check_cnndm.txt", encoding="utf-8"):
        cot_check_prompt += line_2
    for line_3 in open("./prompts/final", encoding="utf-8"):
        cot_fina_prompt += line_3

    prompt = {"std_generation_cnndm_prompt": std_generation_cnndm_prompt,
              "std_generation_xsum_prompt": std_generation_xsum_prompt,
              "cot_sample_prompt": cot_sample_prompt,
              "cot_check_prompt": cot_check_prompt,
              "cot_generation_cnndm_prompt": cot_generation_cnndm_prompt,
              "cot_generation_xsum_prompt": cot_generation_xsum_prompt,
              "cot_extraction_prompt": cot_extraction_prompt,
              "cot_fina_prompt": cot_fina_prompt
              }

    return prompt


def parse_arguments():
    parser = argparse.ArgumentParser(description="SumCoT")
    parser.add_argument("--cot_true", type=bool, default="False",
                        help="standard or cot-based generation")
    parser.add_argument("--model", type=str, default="text-davinci-002",
                        choices=["text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-001", "text-davinci-002", "text-davinci-003"],
                        help="model used for decoding")
    parser.add_argument("--dataset", type=str, default="cnndm",
                        choices=["cnndm", "xsum"], help="dataset source")
    parser.add_argument("--start_id", type=int, default="0")
    parser.add_argument("--end_id", type=int, default="0")
    args = parser.parse_args()

    prompt = get_prompt()
    args.cot = prompt["cot_extraction_prompt"]
    args.cot_sam = prompt["cot_sample_prompt"]
    args.cot_check = prompt["cot_check_prompt"]
    args.cot_fina = prompt["cot_fina_prompt"]

    if args.dataset == "cnndm":
        args.std_prompt = prompt["std_generation_cnndm_prompt"]
        args.cot_prompt = prompt["cot_generation_cnndm_prompt"]
    elif args.dataset == "xsum":
        args.std_prompt = prompt["std_generation_xsum_prompt"]
        args.cot_prompt = prompt["cot_generation_xsum_prompt"]
    else:
        raise "Invalid Dataset!"

    return args
