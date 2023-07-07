import logging
import transformers
import torch
import pandas as pd

from argparse import ArgumentParser
from data.load_dataset import load_test_dataset

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaForCausalLM, 
    LlamaTokenizer,
    GPTNeoXForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed
)


# Set up logging
logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()

def main():

    # set up seed for reproducibility
    set_seed(42)

    # set up argument parser
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, default="0 1")
    parser.add_argument("--model_name", type=str, default="opt medalpaca pythia")
    parser.add_argument("--datasets", type=str, default="Med PubMed MedMC")

    # parse arguments
    args = parser.parse_args()
    prompt = args.prompt.split()
    model_name = args.model_name.split()
    datasets = args.datasets.split()

    # for each prompt method, create a dataframe to store the accuracy
    df_dict = {}
    for prompt_method in prompt:
        df_dict[prompt_method] = pd.DataFrame(index=model_name, columns=datasets)

    # for each prompting method
    for prompt_method in prompt:
        # for each model
        for model_n in model_name:
            # for each dataset
            for dataset in datasets:

                # evaluate the model on the dataset
                accuracy = eval(prompt_method, model_n, dataset)

                # store accuracy in dataframe
                df_dict[prompt_method].loc[model_n, dataset] = accuracy

    # print out the dataframe
    for prompt_method in prompt:
        print(prompt_method + "-shot prompting:")
        print(df_dict[prompt_method])


def eval(prompt_method, model_n, dataset):

    # load model and tokenizer
    if model_n == "opt":
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-max-30b", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-max-30b", use_fast=False)
    elif model_n == "medalpaca":
        model = LlamaForCausalLM.from_pretrained("medalpaca/medalpaca-13b")
        tokenizer = LlamaTokenizer.from_pretrained("medalpaca/medalpaca-13b")
    elif model_n == "pythia":
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-12b-deduped")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped")

    # load dataset
    dataset_path = "../data/processed"
    test_dataset = load_test_dataset(prompt_method, dataset, tokenizer, dataset_path)
    label = test_dataset["label"]
    print("label shape: ", label.shape)
    prompt_length = test_dataset["prompt_length"]

    # set up trainer for evaluation only
    trainer = Trainer(model=model)

    predictions = trainer.predict(test_dataset).predictions
    print("prediction shape: ", predictions.shape)

    # decode predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("decoded prediction shape: ", len(decoded_predictions))

    # extract the answer in the prediction
    answer = []
    for i in range(len(decoded_predictions)):
        output = decoded_predictions[i][prompt_length[i]:]
        # find the first alphabet
        for j in range(len(output)):
            if output[j].isalpha():
                answer.append(output[j])
                break
    
    # calculate accuracy
    correct = 0
    for i in range(len(answer)):
        if answer[i] == label[i]:
            correct += 1
    accuracy = correct / len(answer)

    return accuracy


if __name__ == "__main__":
    main()               
                
 





