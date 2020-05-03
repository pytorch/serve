import transformers
from pathlib import Path
import os
import json
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,
 AutoModelForTokenClassification, AutoConfig)
""" This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
"""
print('Transformers version',transformers.__version__)

def transformers_model_dowloader(mode,pretrained_model_name,num_labels,do_lower_case):
    print("Download model and tokenizer", pretrained_model_name)
    #loading pre-trained model and tokenizer
    if mode== "classification":
        config = AutoConfig.from_pretrained(pretrained_model_name,num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
    elif mode== "question_answer":
        model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
    elif mode== "token_classification":
        config = AutoConfig.from_pretrained(pretrained_model_name,num_labels=num_labels)
        model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)


    # NOTE : for demonstration purposes, we do not go through the fine-tune processing here.
    # A Fine_tunining process based on your needs can be added.
    # An example of Colab notebook for Fine_tunining process has been provided in the README.


    """ For demonstration purposes, we show an example of using question answering
        data preprocessing and inference. Using the pre-trained models will not yeild
        good results, we need to use fine_tuned models. For example, instead of "bert-base-uncased",
        if 'bert-large-uncased-whole-word-masking-finetuned-squad' be passed to
        the AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        a better result can be achieved. Models such as RoBERTa, xlm, xlnet,etc. can be
        passed as the pre_trained models.
    """

    text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """

    questions = [
        "How many pretrained models are available in Transformers?",
        "What does Transformers provide?",
        "Transformers provides interoperability between which frameworks?",
    ]

    device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
    model.to(device)


    for question in questions:
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        for key in inputs.keys():
            inputs[key]= inputs[key].to(device)

        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        answer_start_scores, answer_end_scores = model(**inputs)

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        print(f"Question: {question}")
        print(f"Answer: {answer}\n")


    NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print ("Creation of directory %s failed" % NEW_DIR)
    else:
        print ("Successfully created directory %s " % NEW_DIR)

    print("Save model and tokenizer", pretrained_model_name, 'in directory', NEW_DIR)
    model.save_pretrained(NEW_DIR)
    tokenizer.save_pretrained(NEW_DIR)
    return
if __name__== "__main__":
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'setup_config.json')
    f = open(filename)
    options = json.load(f)
    mode = options["mode"]
    model_name = options["model_name"]
    num_labels = int(options["num_labels"])
    do_lower_case = options["do_lower_case"]
    transformers_model_dowloader(mode,model_name, num_labels,do_lower_case)
