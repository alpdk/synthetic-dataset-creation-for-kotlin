import json
import torch

from tqdm import tqdm
from datasets import load_dataset
from generate_model_answers import clean_answer, StoppingCriteriaSub

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)


def split_problem_data(problem_str):
    # Find the position of the opening and closing triple quotes
    start_index = problem_str.find('"""')
    end_index = problem_str.find('"""', start_index + 3)

    # Check if both quotes were found
    if start_index != -1 and end_index != -1:
        # Extract the code part (up to the start of the comment)
        code = problem_str[:start_index].strip()

        # Extract the comment part (between the triple quotes)
        comment = "/***" + problem_str[start_index + 3:end_index].strip() + "***/\n"

        return code, comment
    else:
        raise ValueError("Comment section not found.")


def generate_kotlin_prompt(model, tokenizer, code, prompt, stop_crit):
    criterion = StoppingCriteriaSub(stops=stop_crit, tokenizer=tokenizer)
    stopping_criteria = StoppingCriteriaList([criterion])

    problem = prompt + code

    problem = tokenizer.encode(problem, return_tensors="pt").to('cuda')

    sample = model.generate(
        problem,
        max_new_tokens=512,
        min_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=4,
        stopping_criteria=stopping_criteria,
    )

    answer = tokenizer.decode(sample[0], skip_special_tokens=True)

    code = clean_answer(answer, 0)
    substring = " {"

    if "fun " not in code:
        return "", ""

    func_head, func_body = code.split(substring, 1)

    func_head = func_head + substring

    return func_head, func_body


def create_synt_data(translate_count = 100, model_name='ibm-granite/granite-3b-code-base-2k', dataset_name="jinaai/code_exercises"):
    new_dataset = {"train": []}

    dataset = load_dataset(dataset_name)['train']

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    len_dataset = len(dataset)
    counter = 0

    for i in tqdm(range(len_dataset)):

        if counter >= translate_count:
            break

        code, comment = split_problem_data(dataset[i]["problem"])

        sol_code = dataset[i]["solution"]

        func_head, func_body = generate_kotlin_prompt(model, tokenizer, comment + '\n' + code + '\n' + sol_code,
                                                      """Complete solution for python code and translate the following Python function to Kotlin.q
                                                      Ensure the Kotlin function has proper formatting with correct indentation.
                                                      The Kotlin function definition should include a `{` immediately after the parameter list, and the rest of the code should be indented as per Kotlin's syntax rules.
                                                      Python code:

                                                      ```python""",
                                                      "\n}\n")

        if func_head == func_body == "":
            continue

        new_dataset["train"].append([])
        new_dataset["train"][-1] = {}

        new_dataset["train"][-1]["prompt"] = comment + "\n\n" + func_head
        new_dataset["train"][-1]["solution"] = func_body

        counter += 1

    with open("test_dataset.json", "w") as outfile:
        json.dump(new_dataset, outfile)
