import sys
import json

model_name = sys.argv[1]

data = {}

if model_name == "en2fr_model":
    data['model_name'] = "TransformerEn2Fr"
    data['translated_output'] = "french_output"
    data['bpe'] = "subword_nmt"
elif model_name == "en2de_model":
    data['model_name'] = "TransformerEn2De"
    data['translated_output'] = "german_output"
    data['bpe'] = "fastbpe"
else:
    data['model_name'] = "TransformerDe2En"
    data['translated_output'] = "english_output"
    data['bpe'] = "fastbpe"

with open('setup_config.json', 'w') as outfile:
    json.dump(data, outfile)
