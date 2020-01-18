import argparse
import logging
import os
import sys
import torch
from model import TextSentiment
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.

    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        cls: a tensor saving the labels of individual text entries.
    """
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_and_valid(lr_, sub_train_, sub_valid_):
    r"""
    We use a SGD optimizer to train the model here and the learning rate
    decreases linearly with the progress of the training process.

    Arguments:
        lr_: learning rate
        sub_train_: the data used to train the model
        sub_valid_: the data used for validation
    """

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_gamma)
    train_data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True,
                            collate_fn=generate_batch, num_workers=args.num_workers)
    num_lines = num_epochs * len(train_data)

    for epoch in range(num_epochs):

        # Train the model
        for i, (text, offsets, cls) in enumerate(train_data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss.backward()
            optimizer.step()
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
                        progress * 100, scheduler.get_lr()[0], loss))
        # Adjust the learning rate
        scheduler.step()

        # Test the model on valid set
        print("")
        print("Valid - Accuracy: {}".format(test(sub_valid_)))


def test(data_):
    r"""
    Arguments:
        data_: the data used to train the model
    """
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)

    # In case that nothing in the dataset
    if total_accuracy == []:
        return 0.0

    return sum(total_accuracy) / len(total_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='num epochs (default=5)')
    parser.add_argument('--embed-dim', type=int, default=32,
                        help='embed dim. (default=32)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--split-ratio', type=float, default=0.95,
                        help='train/valid split ratio (default=0.95)')
    parser.add_argument('--lr', type=float, default=4.0,
                        help='learning rate (default=4.0)')
    parser.add_argument('--lr-gamma', type=float, default=0.8,
                        help='gamma value for lr (default=0.8)')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='ngrams (default=2)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='num of workers (default=1)')
    parser.add_argument('--device', default='cpu',
                        help='device (default=cpu)')
    parser.add_argument('--data', default='.data',
                        help='data directory (default=.data)')
    parser.add_argument('--use-sp-tokenizer', type=bool, default=False,
                        help='use sentencepiece tokenizer (default=False)')
    parser.add_argument('--sp-vocab-size', type=int, default=20000,
                        help='vocab size in sentencepiece model (default=20000)')
    parser.add_argument('--dictionary',
                        help='path to save vocab')
    parser.add_argument('--save-model-path',
                        help='path for saving model')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data = args.data
    split_ratio = args.split_ratio
    # two args for sentencepiece tokenizer
    use_sp_tokenizer = args.use_sp_tokenizer
    sp_vocab_size = args.sp_vocab_size

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(data):
        print("Creating directory {}".format(data))
        os.mkdir(data)

    if use_sp_tokenizer:
        import spm_dataset
        train_dataset, test_dataset = spm_dataset.setup_datasets(args.dataset,
                                                                 root='.data',
                                                                 vocab_size=sp_vocab_size)
        model = TextSentiment(sp_vocab_size, embed_dim,
                              len(train_dataset.get_labels())).to(device)

    else:
        train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
            root=data, ngrams=args.ngrams)
        model = TextSentiment(len(train_dataset.get_vocab()),
                              embed_dim, len(train_dataset.get_labels())).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # split train_dataset into train and valid
    train_len = int(len(train_dataset) * split_ratio)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_and_valid(lr, sub_train_, sub_valid_)
    print("Test - Accuracy: {}".format(test(test_dataset)))

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.state_dict(), args.save_model_path)

    if args.dictionary is not None:
        print("Save vocab to {}".format(args.dictionary))
        torch.save(train_dataset.get_vocab(), args.dictionary)
