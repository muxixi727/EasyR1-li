from mathruler.grader import extract_boxed_content, grade_answer
import re

def math_compute_score(predict_str: str, ground_truth: str, max_length: int) -> float:
    return score(predict_str, ground_truth, max_length)
    answer = extract_boxed_content(predict_str)
    if answer == "None":
        return 0.0  # no answer

    if grade_answer(answer, ground_truth):
        return 1.0  # correct answer

    return 0.1  # wrong answer

def score(predict_str: str, ground_truth: str, max_length: int) -> float:
    for_reward = format_score(predict_str)
    if for_reward == 0.0:
        acc_reward = 0.0
    else:
       acc_reward = accuracy_score(predict_str, ground_truth)
    cos_reward = cosine_score(predict_str, acc_reward, max_length)

    score_all = for_reward + acc_reward + cos_reward
    return score_all

def cosfn(t, T, min_value, max_value):
    import math
    return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

def cosine_score(predict_str: str, acc_reward: float, max_length: int) -> float:
    is_correct = acc_reward >= 1
    if is_correct:
        # 对于正确答案，交换 min/max
        min_value = 0.5
        max_value = 1.0
    else:
        # 对于错误答案，使用错误范围
        min_value = 0.0
        max_value = -0.5

    gen_len = len(predict_str)  # 获取生成文本的长度
    reward = cosfn(gen_len, max_length, min_value, max_value)  # 计算基于长度的奖励
    return reward

def accuracy_score(predict_str: str, ground_truth: str) -> float:
    sol_match = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
    answer = sol_match.group(1).strip()
    if ground_truth == "A":
        ground_truth = "异常"
    else:
        ground_truth = "正常"
    # Check if the answer matches the ground truth
    if answer == ground_truth:
        return 1.0
    elif ground_truth in answer:
        return 0.3
    else:
        return 0.0

def format_score(predict_str: str) -> float:
    pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
    match = re.match(pattern, predict_str, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0
