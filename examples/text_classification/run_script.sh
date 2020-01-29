if [ ! -d ".data" ]; then
    mkdir .data
fi

python train.py AG_NEWS --device cpu --save-model-path  model.pt --dictionary source_vocab.pt