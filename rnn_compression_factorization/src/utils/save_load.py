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

import torch

def save_model(model,path="./weights",name="comp_vmmodel_wNone_uNone"):
    torch.save(model.state_dict(),path+name+".pkl")

def load_model(model,path="./weights",name="comp_vmmodel_wNone_uNone"):
    state_dict=torch.load(path+name+".pkl")
    model.load_state_dict(state_dict)
    return model