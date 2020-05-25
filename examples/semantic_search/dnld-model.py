from sentence_transformers import SentenceTransformer

xformer = SentenceTransformer('bert-base-nli-mean-tokens')
xformer.save('/tmp/sentence_transformer')
