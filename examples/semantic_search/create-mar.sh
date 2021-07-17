#!/bin/bash

tmp_dir=/tmp/sentence_transformer

python dnld-model.py

mar_path=`pwd`
cd $tmp_dir
zip -r $mar_path/bert.pt 0_BERT
zip -r $mar_path/pool.zip 1_Pooling
cd -
cp $tmp_dir/config.json $tmp_dir/modules.json .

torch-model-archiver --model-name sentence_xformer --version 1.0 --serialized-file bert.pt --handler ./semantic_search_handler.py --extra-files "./config.json,./pool.zip,./modules.json"