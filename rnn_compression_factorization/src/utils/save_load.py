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
