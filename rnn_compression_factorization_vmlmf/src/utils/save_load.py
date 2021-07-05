import os
import sys
import torch

def save_model(model,args,path="./train_test/trained_model/",name=None):
    
    name="comp_vmmodel_wRank{}_uRank_{}_data_{}_layer{}_seed{}".format(args.wRank,args.uRanks,args.data,args.layer_sizes,args.seed) if name is None else name
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),path+name+".pkl")
    print("model saved in %s"%(path+name+".pkl"))

def load_model(model,args,path="./train_test/trained_model/",name=None):
    
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
