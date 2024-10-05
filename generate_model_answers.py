import re
import sys
import torch
import jsonlines

from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer):
        (StoppingCriteria.__init__(self),)
        self.stops = rf"{stops}"
        self.tokenizer = tokenizer

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_three_tokens = [int(x) for x in input_ids.data[0][-3:]]
        decoded_last_three_tokens = self.tokenizer.decode(last_three_tokens)

        return bool(re.search(self.stops, decoded_last_three_tokens))


def generate(problem, model, tokenizer):
    criterion = StoppingCriteriaSub(stops="\n}\n", tokenizer=tokenizer)
    stopping_criteria = StoppingCriteriaList([criterion])

    problem = tokenizer.encode(problem, return_tensors="pt").to('cuda')

    sample = model.generate(
        problem,
        max_new_tokens=256,
        min_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=1,
        stopping_criteria=stopping_criteria,
    )

    answer = tokenizer.decode(sample[0], skip_special_tokens=True)
    return answer


def clean_answer(code, skip_lines=1):
    # Clean comments
    code_without_line_comments = re.sub(r"//.*", "", code)

    code_without_all_comments = re.sub(
        r"/\*.*?\*/", "", code_without_line_comments, flags=re.DOTALL
    )

    # Clean signatures
    lines = code_without_all_comments.split("\n")

    for i, line in enumerate(lines):
        if line.startswith("fun "):
            return "\n".join(lines[i + skip_lines:])

    return code


def generate_model_answers(model_name="ibm-granite/granite-3b-code-base-2k",
                           dataset_name="jetbrains/Kotlin_HumanEval"):

    dataset = load_dataset(dataset_name)['train']
    problem_dict = {problem['task_id']: problem for problem in dataset}

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output = []
    for key in tqdm(list(problem_dict.keys()), leave=False):
        problem = problem_dict[key]["prompt"]
        answer = generate(problem, model, tokenizer)
        answer = clean_answer(answer)
        output.append({"task_id": key, "completion": answer, "language": "kotlin"})

    output_file = f"answers"

    with jsonlines.open(output_file, mode="w") as writer:
        for line in output:
            writer.write(line)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        generate_model_answers()
    elif len(sys.argv) == 2:
        generate_model_answers(sys.argv[1])
    elif len(sys.argv) == 3:
        generate_model_answers(sys.argv[1], sys.argv[2])
    else:
        print("Wrong number of arguments")