## train 과정에서 
"""
path="./weights/"
name="comp_vmmodel_w{}_u{}".format(args.wRank,args.uRanks)
state_dict=torch.load(path+name+".pkl")
model.load_state_dict(state_dict)
print_model_parm_nums(model)
print_model_parm_names(model)
total_ops=count_lstm(model,args.max_steps,args.batch_size)
total_ops+=count_linear(model,10)
print("+ total_ops:",total_ops)
"""
#
import os
import sys
import torch

def save_model(model,path="./trained_model/",name="comp_vmmodel_wNone_uNone"):
    
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),path+name+".pkl")
    print("model saved in %s"%(path+name+".pkl"))

def load_model(model,path="./trained_model/",name="comp_vmmodel_wNone_uNone"):
    
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
