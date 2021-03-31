# Workflow pipeline example using nmt transformer nlp model

This example uses the existing [nmt_transformers](../../nmt_transformer) standalone example to create a workflow. We use two models, the English to German translator and the German to English translator to demonstrate stringing models together in a sequential flow.

The model flow is composed of:
- TransformerEn2De (Workflow model): Transformer model which translates the English text to German
- Intermediate-input-processing (Workflow function): Strips the input key from the previous model output and passing the translated text as expected to the next model
- TransformerDe2En (Workflow model): Transformer model which translates the German text back to English
- Post-processing (Workflow function): Maps the output to show both the German translation and the English re-translation

## Flow

  Input Text -> TransformerEn2De -> Intermediate-input-processing -> TransformerDe2En -> Post-processing

## Commands to create the models and the workflow
```
cd $TORCH_SERVE_DIR/examples/nmt_transformer/
./create_mar.sh de2en_model
./create_mar.sh en2de_model

cd $TORCH_SERVE_DIR/examples/Workflows/nmt_tranformers_pipeline/
mkdir model_store wf_store
mv $TORCH_SERVE_DIR/examples/nmt_transformer/model_store/*.mar model_store/
torch-workflow-archiver -f --workflow-name nmt_wf --spec-file nmt_workflow.yaml --handler nmt_workflow_handler.py --export-path wf_store/
```

## Serve the workflow
```
> torchserve --start --model-store model_store/ --workflow-store wf_store/ --ncs --ts-config config.properties
> curl -X POST "http://127.0.0.1:8081/workflows?url=nmt_wf.war"
{
  "status": "Workflow nmt_wf has been registered and scaled successfully."
}

# Single input
> curl http://127.0.0.1:8080/wfpredict/nmt_wf -T model_input/sample.txt
{
  "german_translation": "Hallo James, wann kommst du nach Hause? Ich warte auf dich. Bitte komm so bald wie m\u00f6glich.",
  "english_re_translation": "Hi James, when are you coming home? I am waiting for you. Please come as soon as possible."
}

# Batched input
> curl http://127.0.0.1:8080/wfpredict/nmt_wf -T model_input/sample1.txt& \
> curl http://127.0.0.1:8080/wfpredict/nmt_wf -T model_input/sample2.txt& \
> curl http://127.0.0.1:8080/wfpredict/nmt_wf -T model_input/sample3.txt& \
> curl http://127.0.0.1:8080/wfpredict/nmt_wf -T model_input/sample4.txt&
{
  "german_translation": "Hallo Welt!!!",
  "english_re_translation": "Hello world!!!"
}{
  "german_translation": "Es tut mir leid, ich erinnere mich nicht an Ihren Namen. Sie sind es?",
  "english_re_translation": "I'm sorry, I don't remember your name. Is it you?"
}{
  "german_translation": "Mir geht es gut. Wie geht es Ihnen? Es l\u00e4uft gut, danke. Wie geht es Ihnen? Gut, danke. Und sich selbst?",
  "english_re_translation": "How are you? It's going well, thank you. How are you? Good, thank you. And yourself?"
}{
  "german_translation": "Hallo James, wann kommst du nach Hause? Ich warte auf dich. Bitte komm so bald wie m\u00f6glich.",
  "english_re_translation": "Hi James, when are you coming home? I am waiting for you. Please come as soon as possible."
}

# Unregister workflow
> curl -X DELETE "http://127.0.0.1:8081/workflows/nmt_wf"
{
  "status": "Workflow \"nmt_wf\" unregistered"
}
```
