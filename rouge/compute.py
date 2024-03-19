from rouge import Rouge
from bert_score import score


if __name__ == '__main__':
    # 读取文本摘要和参考文本
    with open('output', 'r', encoding='utf-8') as f1, \
         open('test.tgt', 'r', encoding='utf-8') as f2:
        # generated_text = f1.read()
        # reference_texts = f2.read()

        rouge1, rouge2, rougel = 0, 0, 0  # 初始化 ROUGE 指标的值
        bert_score = 0  # 初始化 BERTScore 的值
        count = 0

        # 计算 ROUGE 指标
        rouge = Rouge()

        for generated_text, reference_text in zip(f1, f2):
            scores = rouge.get_scores(generated_text, reference_text)

            rouge1 += scores[0]["rouge-1"]["f"]
            rouge2 += scores[0]["rouge-2"]["f"]
            rougel += scores[0]["rouge-l"]["f"]

            # 计算 BERTScore
            _, _, bert_score = score([generated_text], [reference_text], lang="others", verbose=True)

            count += 1

        # 计算平均值
        rouge1_avg = rouge1 / count
        rouge2_avg = rouge2 / count
        rougel_avg = rougel / count
        bert_score_avg = bert_score.mean().item()

        # 打印结果
        print("ROUGE-1", rouge1_avg)
        print("ROUGE-2", rouge2_avg)
        print("ROUGE-L", rougel_avg)
        print("BERTScore", bert_score_avg)
