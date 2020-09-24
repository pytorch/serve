# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import json

separator = "="
keys = {}

# I named your file conf and stored it 
# in the same directory as the script

with open('/home/kc/config.properties') as f:

    for line in f:
        if separator in line:

            # Find the name and value by splitting the string
            name, value = line.split(separator, 1)

            # Assign key value pair to dict
            # strip() removes white space from the ends of strings
            keys[name.strip()] = value.strip()
            

    
keys['model_snapshot'] = json.loads(keys['model_snapshot'])

model = keys['model_snapshot']['models']