# import sys
# import os
# import json
#
# from datasets import load_dataset
#
# # Get the path to the mxeval directory
# current_dir = sys.argv[1]
# mxeval_dir = os.path.join(current_dir, 'mxeval')
#
# # Add mxeval to sys.path
# if sys.path[0] != mxeval_dir:
#
#     if mxeval_dir in sys.path:
#         sys.path.remove(mxeval_dir)
#
#     sys.path.insert(0, mxeval_dir)
#
# # Print all system path for packages
# print(sys.path, sep='\n')
#
# from mxeval.evaluation import evaluate_functional_correctness
#
# def main():
#     dataset = load_dataset("jetbrains/Kotlin_HumanEval")['train']
#     problem_dict = {problem['task_id']: problem for problem in dataset}
#
#     output_file = sys.argv[2]
#
#     evaluate_functional_correctness(
#         sample_file=output_file,
#         k=[1],
#         n_workers=16,
#         timeout=15,
#         problem_file=problem_dict,
#     )
#
#     with open(output_file + '_results.jsonl') as fp:
#         total = 0
#         correct = 0
#
#         for line in fp:
#             sample_res = json.loads(line)
#             print(sample_res)
#             total += 1
#             correct += sample_res['passed']
#
#         print(f'Pass rate: {correct/total}')
#
# if __name__ == "__main__":
#     main()

import sys
import os
import json

from datasets import load_dataset
from setup_mxeval import setup_mxeval

setup_mxeval()

from mxeval.evaluation import evaluate_functional_correctness

# Give the percentage of successfully generated kotlin code blocks
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

check_kotlin_code()
