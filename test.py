# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# from optimum.bettertransformer import BetterTransformer

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# model = BetterTransformer.transform(model)

# dummy_input = "translate to German: Hello, my name is Wolfgang and I live in Berlin."
# dummy_input = tokenizer(dummy_input, return_tensors="pt")

# output = model.generate(**dummy_input)
# print(tokenizer.decode(output[0], skip_special_tokens=True))


import logging
import os

logger = logging.getLogger(__name__)

# assume model_yaml_config is already loaded from a YAML file
model_yaml_config = {"handler": {"some_other_key": "some_other_value"}}

try:
    index_filename = model_yaml_config["handler"]["index_filename"]
    logger.info("Found index_filename in config: %s", index_filename)
except KeyError:
    index_filename = None
    logger.warning("index_filename not found in config, setting to None")

assert os.path.exists(index_filename), f"Index file '{index_filename}' not found"
# use index_filename in your code
