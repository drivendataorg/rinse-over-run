# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 01:22:58 2019

@author: vt
"""

import json
from model import build_model

if __name__ == "__main__":
    #read json files
    read_json = json.loads(open("config.json").read())
    #build model and create submission
    build_model(read_json["train_set"],read_json["test_set"],read_json["label_file"],read_json['sample_submission_file'],
           read_json["recipe_metadata"],read_json['final_submission_path'])
    
    
    