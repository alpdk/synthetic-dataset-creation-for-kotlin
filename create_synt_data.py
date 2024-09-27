import torch

from datasets import load_dataset
from generate_base_model_answers import StoppingCriteriaSub, clean_answer

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
        max_new_tokens=256,
        min_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=1,
        stopping_criteria=stopping_criteria,
    )

    answer = tokenizer.decode(sample[0], skip_special_tokens=True)

    code = clean_answer(answer, 0)
    substring = " {"

    func_head, func_body = code.split(substring, 1)

    func_head = func_head + substring

    return func_head, func_body


def create_synt_data(model_name='ibm-granite/granite-3b-code-base-2k', dataset_name="jinaai/code_exercises"):
    dataset = load_dataset(dataset_name)['train']

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Print the dataset structure
    # print(dataset)

    code, comment = split_problem_data(dataset[100]["problem"])

    func_head, func_body = generate_kotlin_prompt(model, tokenizer, code,
                                                  "Translate code from python to kotlin, also head of the function should contain {",
                                                  "\n}\n")

    print(func_head, end='\n\n')
    print(func_body, end='\n\n')


create_synt_data()
