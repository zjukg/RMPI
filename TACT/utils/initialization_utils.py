import os
import logging
import json
import torch


def initialize_experiment(params):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'expri_save_models')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.expri_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)


    print('============ Params ============')
    print('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    print('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, model, load_model=False):
    '''
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    '''

    if load_model and os.path.exists(os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        print('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
        graph_classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth')).to(device=params.device)
    else:
        relation2id_path = os.path.join(params.main_dir, f'../data/{params.dataset}/relation2id.json')
        with open(relation2id_path) as f:
            relation2id = json.load(f)

        print('No existing model found. Initializing new model..')
        graph_classifier = model(params, relation2id).to(device=params.device)

    return graph_classifier
