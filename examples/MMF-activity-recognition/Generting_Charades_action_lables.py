import sys
import os
import pandas as pd
from pathlib import Path

os.system('wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip')
os.system('wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip')
os.system('unzip Charades_v1_480.zip')
os.system('unzip Charades.zip')

def make_charades_df(csv_path, video_dir, classes_file):
    # load the csv
    df = pd.read_csv(csv_path)

    # transform the id to a pathname
    df['path'] = df['id'].map(lambda x: '{}{}.mp4'.format(video_dir, x))

    # parse action labels
    df['action_labels'] = df['actions'].map(
        lambda x: [l.split(' ')[0] for l in x.split(';')] if pd.notnull(x) else []
    )

    # load id to class map
    with open(classes_file, 'r') as f:
        class_names = f.readlines()

    id2classname = {}
    for c in class_names:
        class_id = c.split(' ')[0]
        class_name = ' '.join(c.split(' ')[1:]).strip('\n')
        id2classname[class_id] = class_name

    # transform label ids to names
    df['action_labels'] = df['action_labels'].map(
        lambda x:[id2classname[class_id] for class_id in x]
    )

    # filter only these videos that actually exist
    df_exists = df[df['path'].map(lambda x: Path(x).exists())]

video_dir = 'Charades_v1_480/'
charades_classes_file = 'Charades/Charades_v1_classes.txt'
train_csv_path = 'Charades/Charades_v1_train.csv'
# we get the df for training
df_train = make_charades_df(
    csv_path=train_csv_path,
    video_dir=video_dir,
    classes_file=charades_classes_file
)

df['action_labels'].to_csv("./charades_action_lables.csv")
