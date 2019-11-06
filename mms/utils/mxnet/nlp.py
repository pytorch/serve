# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
NLP utils
"""
import bisect
import numpy as np
import mxnet as mx


def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='\n', start_label=0):
    """Encode sentences and (optionally) build a mapping
    from string tokens to integer indices. Unknown keys
    will be added to vocabulary.

    Parameters
    ----------
    sentences : list of list of str
        A list of sentences to encode. Each sentence
        should be a list of string tokens.
    vocab : None or dict of str -> int
        Optional input Vocabulary
    invalid_label : int, default -1
        Index for invalid token, like <end-of-sentence>
    invalid_key : str, default '\\n'
        Key for invalid token. Use '\\n' for end
        of sentence by default.
    start_label : int
        lowest index.

    Returns
    -------
    result : list of list of int
        encoded sentences
    vocab : dict of str -> int
        result vocabulary
    """
    idx = start_label
    if vocab is None:
        vocab = {invalid_key: invalid_label}
        new_vocab = True
    else:
        new_vocab = False
    res = []
    for sent in sentences:
        coded = []
        for word in sent:
            if word not in vocab:
                if not new_vocab:
                    coded.append(invalid_label)
                    continue
                else:
                    if idx == invalid_label:
                        idx += 1
                    vocab[word] = idx
                    idx += 1
            coded.append(vocab[word])
        res.append(coded)

    return res, vocab


def pad_sentence(sentence, buckets, invalid_label=-1, data_name='data', layout='NT'):
    """Pad a sentence to closest length in provided buckets.

        Parameters
        ----------
        sentence : list of int
            A list of integer representing an encoded sentence.
        buckets : list of int
            Size of the data buckets.
        invalid_label : int, optional
            Index for invalid token, like <end-of-sentence>.
        data_name : str, optional
            Input data name.
        layoutlayout : str, optional
            Format of data and label. 'NT' means (batch_size, length)
            and 'TN' means (length, batch_size).

        Returns
        -------
        result : mx.io.DataBatch
            DataBatch contains sentence.
        """
    buck = bisect.bisect_left(buckets, len(sentence))
    buff = np.full((buckets[buck],), invalid_label, dtype='float32')
    buff[:len(sentence)] = sentence
    sent_bucket = buckets[buck]
    pad_sent = mx.nd.array([buff], dtype='float32')
    shape = (1, sent_bucket) if layout == 'NT' else (sent_bucket, 1)
    return mx.io.DataBatch([pad_sent], pad=0, bucket_key=sent_bucket,
                           provide_data=[mx.io.DataDesc(
                               name=data_name,
                               shape=shape,
                               layout=layout)])
