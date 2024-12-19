from sanity_utils import validate_model_on_gpu

model_loaded = validate_model_on_gpu()

if not model_loaded:
    exit(1)
