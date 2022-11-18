# Steps

```
$ cd serve
$ torch-model-archiver --model-name tokenizer --version 1.0 \
    --serialized-file cpp/test/resources/torchscript_model/text_classifier/text_classifier_handler/tokenizer.pt \
    --handler examples/text_classification_with_scriptable_tokenizer/bert_example/handler.py \
    --extra-files examples/text_classification_with_scriptable_tokenizer/index_to_name.json
$ mv tokenizer.pt model_store/
$ torchserve --start --model-store model_store --models my_tc=tokenizer.mar
$ curl http://127.0.0.1:8080/predictions/my_tc -T cpp/test/resources/torchscript_model/text_classifier/test_input.txt
```
