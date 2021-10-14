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
import os
import sys
import torch

def save_model(model,args,path="./train_test/trained_model/",name=None):
    """
    Save the compressed state dictionary of model to input file path
    @param model
        model to save
    @param args
        argument user decided
    @param path
        path to save the model in
    @param name
        filename to save
    """
    
    name="comp_vmmodel_wRank{}_uRank_{}_data_{}_layer{}_seed{}".format(args.wRank,args.uRanks,args.data,args.layer_sizes,args.seed) if name is None else name
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),path+name+".pkl")
    print("model saved in %s"%(path+name+".pkl"))

def load_model(model,args,path="./train_test/trained_model/",name=None):
    """
    load the compressed model from input file path
    @param model
        model to load
    @param args
        argument user decided
    @param path
        path where the model exists
    @param name
        filename to load
    """

    name="comp_vmmodel_wRank{}_uRank_{}_data_{}_layer{}_seed{}".format(args.wRank,args.uRanks,args.data,args.layer_sizes,args.seed) if name is None else name
    file=path+name+".pkl"
    
    if os.path.exists(file):
        state_dict=torch.load(file)
        model.load_state_dict(state_dict)
        print("model restored from %s"%(file))
    else:
        print(name+'pkl does not exists.')
        print("Testing can only be done when the trained model exists.")
        sys.exit()
        
    return model       
