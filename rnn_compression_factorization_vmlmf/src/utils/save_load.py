################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: save_load.py
# - utilities for save and load compressed model
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102,C0413, E0401
"""
====================================
 :mod:`save_load`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
모델을 저장하고, 저장한 모델을 불러오기 위한 모듈입니다.

"""
import os
import sys
import torch

def save_model(model,args,path="./train_test/trained_model/",name=None):
    """Save the compressed state dictionary of model to input file path

    :param model: model to save
    :param args: argument user decided
    :param path: string path to save the model in
    :param name: string filename to save
    """

    name=f"comp_vmmodel_wRank{args.wRank}_uRank_{args.uRanks}_data_{args.data}\
        _layer{args.layer_seed}_seed{args.seed}" if name is None else name
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),path+name+".pkl")
    print(f"model saved in {path+name}.pkl")

def load_model(model,args,path="./train_test/trained_model/",name=None):
    """load the compressed model from input file path

    :param model: model to load
    :param args: argument user decided
    :param path: string file path where the model exists
    :param name: string filename to load
    """

    name=f"comp_vmmodel_wRank{args.wRank}_uRank_{args.uRanks}_data_{args.data}\
        _layer{args.layer_seed}_seed{args.seed}" if name is None else name
    file=path+name+".pkl"

    if os.path.exists(file):
        state_dict=torch.load(file)
        model.load_state_dict(state_dict)
        print("model restored from {file}")
    else:
        print(name+'pkl does not exists.')
        print("Testing can only be done when the trained model exists.")
        sys.exit()

    return model
