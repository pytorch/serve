# Handler for KServe v2 protocol

## For KServe v2 protocol the tensor output should be converted to list of list with `[output.flatten().tolist()]` from the postprocess method in the handler.

### Refer: [mnist handler](./mnist_handler.py) and [bert_handler](./bert_handler.py)
