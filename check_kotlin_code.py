import sys
import json

from datasets import load_dataset
from setup_mxeval import setup_mxeval

setup_mxeval()

from mxeval.evaluation import evaluate_functional_correctness

def check_kotlin_code(file_to_check = 'answers'):
    dataset = load_dataset("jetbrains/Kotlin_HumanEval")['train']
    problem_dict = {problem['task_id']: problem for problem in dataset}

    evaluate_functional_correctness(
        sample_file=file_to_check,
        k=[1],
        n_workers=16,
        timeout=15,
        problem_file=problem_dict,
    )

    with open(file_to_check + '_results.jsonl') as fp:
        total = 0
        correct = 0

        for line in fp:
            sample_res = json.loads(line)
            print(sample_res)
            total += 1
            correct += sample_res['passed']

        print(f'Pass rate: {correct/total}')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_kotlin_code()
    elif len(sys.argv) == 2:
        check_kotlin_code(sys.argv[1])
    else:
        print("Wrong number of arguments")