**Deploy QnA elgeish/cs224n-squad2.0-albert-base-v2 model to torchserve

1. Edit hf_question_answer_classifier.py file and specify `NEW_DIR` value

2. download and save pretrained elgeish/cs224n-squad2.0-albert-base-v2 for question-answer transformation
`python hf_question_answer_classifier.py`

3. create mar file for deployment
`torch-model-archiver --model-name bert-qa --version 1.0 --serialized-file ./pytorch_model.bin --handler ./hf_question_answer_classifier.py --extra-files "./config.json,./spiece.model,./tokenizer_config.json,./special_tokens_map.json"`

4. copy the generted bert-qa.mar to your model-store directory

5. start torchserve as
`torchserve --start --model-store /Users/dhaniram_kshirsagar/projects/neo-sagemaker/mms/model-store --models bert-qa=bert-qa.mar`

6. do inference with sample_text.txt file
`curl http://127.0.0.1:8080/predictions/bert-qa -T sample_text.txt `
