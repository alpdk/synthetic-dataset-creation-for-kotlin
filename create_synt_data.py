from datasets import load_dataset

def split_problem_data(problem_str):
    # Find the position of the opening and closing triple quotes
    start_index = problem_str.find('"""')
    end_index = problem_str.find('"""', start_index + 3)

    # Check if both quotes were found
    if start_index != -1 and end_index != -1:
        # Extract the code part (up to the start of the comment)
        code = problem_str[:start_index].strip()

        # Extract the comment part (between the triple quotes)
        comment = problem_str[start_index + 3:end_index].strip()

        return code, comment
    else:
        raise ValueError("Comment section not found.")


def create_synt_data(dataset_name = "jinaai/code_exercises"):
    # Load the dataset
    dataset = load_dataset("jinaai/code_exercises")

    # Print the dataset structure
    print(dataset)

    # Accessing the training set
    train_dataset = dataset['train']

    # Display the first example in the training set
    # print(train_dataset[100]["problem"])

    code, comment = split_problem_data(train_dataset[100]["problem"])

    print("Code: ", code, end='\n')
    print("Сomment: ", comment, end='\n')

create_synt_data()