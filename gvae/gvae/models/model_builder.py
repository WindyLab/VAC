from . import gvae_model
import os
import pdb

def build_model(params):
    model = gvae_model.GVAE(params)
    print("GVAE MODEL: ",model)
    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    if params['gpu']:
        model.cuda()
    return model

