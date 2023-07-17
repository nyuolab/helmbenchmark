from datasets import load_dataset
import os.path as path
import torch

def load_test_dataset(prompting, d_name, tokenizer, dataset_path):

    # dataset name dictionary
    dataset_name = {
        "Med": "GBaker/MedQA-USMLE-4-options",
        "PubMed": "bigbio/pubmed_qa",
        "MedMC": "medmcqa"
    }

    # load dataset
    pubmed_subset = "pubmed_qa_labeled_fold0_source"
    if d_name == "PubMed":
        dataset = load_dataset(dataset_name[d_name], name=pubmed_subset)
    else:
        dataset = load_dataset(dataset_name[d_name])

    if prompting != 0:
        # the instruction
        instruction = "The following are multiple choice questions (with answers) about medical knowledge.\n\n"
    else:
        # the instruction
        instruction = ""

    if path.exists(path.join(dataset_path, d_name + ".pt")):
        # load the preprocessed dataset
        test_dataset = torch.load(path.join(dataset_path, d_name + ".pt"))
    else:
        # preprocess the dataset
        test_dataset = globals()["process_" + dataset_name[d_name].split("/")[-1].replace("-", "_")](prompting, instruction, dataset)
        # save the preprocessed dataset
        torch.save(test_dataset, path.join(dataset_path, d_name + ".pt"))
    
    # tokenize the dataset
    def tokenize(examples):
        tokenized = tokenizer(examples["prompt"])
        tokenized["truth"] = examples["label"]
        tokenized["prompt_length"] = examples["prompt_length"]
        return tokenized
    
    if path.exists(path.join(dataset_path, d_name + "_" + tokenizer.name_or_path.replace('/', '_') + ".pt")):
        # load the tokenized dataset
        tokenized_test_dataset = torch.load(path.join(dataset_path, d_name + "_" + tokenizer.name_or_path.replace('/', '_') + ".pt"))
    else:
        # preprocess the dataset
        tokenized_test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)
        # save the preprocessed dataset
        torch.save(tokenized_test_dataset, path.join(dataset_path, d_name + "_" + tokenizer.name_or_path.replace('/', '_') + ".pt"))
    
    return tokenized_test_dataset

# function for pre-processing the MedQA dataset
def process_MedQA_USMLE_4_options(prompting, instruction, dataset):
    
    # the prompt template starts with the instruction
    template = instruction

    # add a few examples to the template
    for i in range(prompting):
        template = (template +
                   "Question: " + dataset["train"][i]["question"] + "\n" +
                   "(A) " + dataset["train"][i]["options"]['A'] + " " +
                   "(B) " + dataset["train"][i]["options"]['B'] + " " +
                   "(C) " + dataset["train"][i]["options"]['C'] + " " +
                   "(D) " + dataset["train"][i]["options"]['D'] + "\n" +
                   "Answer: (" + dataset["train"][i]["answer_idx"] + ")\n\n")
    
    # function for converting the dataset into prompt-label format
    def prompt_label(example):
        example["prompt"] = (
            template +
            "Question: " + example["question"] + "\n" +
            "(A) " + example["options"]['A'] + " " +
            "(B) " + example["options"]['B'] + " " +
            "(C) " + example["options"]['C'] + " " +
            "(D) " + example["options"]['D'] + "\n" +
            "Answer:"
        )
        example["label"] = example["answer_idx"]
        example["prompt_length"] = len(example["prompt"])
        return example
    
    test_dataset = dataset["test"].map(prompt_label, remove_columns=dataset["test"].column_names)

    return test_dataset

# function for pre-processing the PubMedQA dataset
def process_pubmed_qa(prompting, instruction, dataset):

    # the prompt template starts with the instruction
    template = instruction
    
    # dictionary for converting the answer to the corresponding option
    answer_dict = {
        'yes': 'A',
        'no': 'B',
        'maybe': 'C'
    }

    # add a few examples to the template
    for i in range(prompting):
        template = (template +
                    "Answer the following question given the context (reply with one of the options): " +
                    "Context: " + ' '.join(dataset["train"][i]["CONTEXTS"]) + ' ' +
                    "Question: " + dataset["train"][i]["QUESTION"] + "\n" +
                    "(A) Yes (B) No (C) Maybe\n" +
                    "Answer: (" + answer_dict[dataset["train"][i]["final_decision"]] + ")\n\n")
        
    # function for converting the dataset into prompt-label format
    def prompt_label(example):
        example["prompt"] = (
            template +
            "Answer the following question given the context (reply with one of the options): " +
            "Context: " + ' '.join(example["CONTEXTS"]) + ' ' +
            "Question: " + example["QUESTION"] + "\n" +
            "(A) Yes (B) No (C) Maybe\n" +
            "Answer:"
        )
        example["label"] = answer_dict[example["final_decision"]]
        example["prompt_length"] = len(example["prompt"])
        return example
    
    test_dataset = dataset["test"].map(prompt_label, remove_columns=dataset["test"].column_names)

    return test_dataset

# function for pre-processing the MedMCQA dataset
def process_medmcqa(prompting, instruction, dataset):

    # the prompt template starts with the instruction
    template = instruction

    # list for converting the answer to the corresponding option
    answer_list = ['A', 'B', 'C', 'D']

    # add a few examples to the template
    for i in range(prompting):
        template = (template +
                    "Question: " + dataset["train"][i]["question"] + "\n" +
                    "(A) " + dataset["train"][i]["options"]['opa'] + " " +
                    "(B) " + dataset["train"][i]["options"]['opb'] + " " +
                    "(C) " + dataset["train"][i]["options"]['opc'] + " " +
                    "(D) " + dataset["train"][i]["options"]['opd'] + "\n" +
                    "Answer: (" + answer_list[dataset["train"][i]["cop"]] + ")\n\n")
        
    # function for converting the dataset into prompt-label format
    def prompt_label(example):
        example["prompt"] = (
            template +
            "Question: " + example["question"] + "\n" +
            "(A) " + example["options"]['opa'] + " " +
            "(B) " + example["options"]['opb'] + " " +
            "(C) " + example["options"]['opc'] + " " +
            "(D) " + example["options"]['opd'] + "\n" +
            "Answer:"
        )
        example["label"] = answer_list[example["cop"]]
        example["prompt_length"] = len(example["prompt"])
        return example
    
    test_dataset = dataset["validation"].map(prompt_label, remove_columns=dataset["validation"].column_names)

    return test_dataset

    


        