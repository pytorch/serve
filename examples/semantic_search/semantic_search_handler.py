"""
Handler for semantic_search using SentenceTransformer 
"""

from sentence_transformers import SentenceTransformer
import scipy.spatial
import json
import zipfile

class SematicSearch(object):
    """
    SematicSearch handler class. This handler takes a corpus and query strings
    as input and returns the closest 5 sentences of the corpus for each query sentence based on cosine similarity.
    Ref - https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search.py
    """

    def __init__(self):
        super(SematicSearch, self).__init__()
        self.initialized = False
        self.embedder = None
    
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(model_dir)
        with zipfile.ZipFile(model_dir + '/bert.pt', 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        with zipfile.ZipFile(model_dir + '/pool.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        self.embedder = SentenceTransformer(model_dir)
        self.initialized = True
    
    def preprocess(self, data):
        print(data)
        inputs = data[0].get("data")
        print(inputs)
        if inputs is None:
            inputs = data[0].get("body")
        inputs = inputs.decode('utf-8')
        inputs = json.loads(inputs)
        corpus = inputs['corpus']
        queries = inputs['queries']

        return corpus, queries

    def inference(self, data):
        corpus_embeddings = self.embedder.encode(data[0])
        query_embeddings = self.embedder.encode(data[1])
	
        closest_n = 5
        inf_data = []
        for query, query_embedding in zip(data[1], query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            qry_result = []
            for idx, distance in results[0:closest_n]:
                qry_result.append(data[0][idx].strip()+" (Score: %.4f)" % (1-distance))
            inf_data.append({"Query":query, "Results":qry_result})

        return [inf_data]

    def postprocess(self, data):
        return data


_service = SematicSearch()

def handle(data, context):
    """
    Entry point for SematicSearch handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise Exception("Unable to process input data. " + str(e))

# Tester
'''class Ctx(object):
    pass

corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]

queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

data = {'corpus':corpus,'queries':queries}

properties = {}
properties["model_dir"] = '/Users/dhaniram_kshirsagar/projects/neo-sagemaker/mms/code/serve/examples/semantic_search'
ctx = Ctx( )
ctx.system_properties = properties
output = handle([{'data':json.dumps(data)}],ctx)

print(output)'''
