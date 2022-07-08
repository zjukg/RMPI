# from comet_ml import Experiment
import pdb
import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning
import numpy as np

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from managers.evaluator import Evaluator

from warnings import simplefilter
import random
import torch.nn as nn

def process_files(files, saved_relation2id):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}

    if saved_relation2id is None:
        relation2id = {}
        rel = 0
    else:
        relation2id = saved_relation2id
        rel = len(saved_relation2id.keys())

    triplets = {}

    ent = 0
    # rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}



    return triplets, entity2id, relation2id, id2entity, id2relation
def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)



    graph_classifier = initialize_model(params, None, load_model=True)
    ori_rels_num = len(graph_classifier.relation2id.keys())
    copyed_rel_embed = graph_classifier.rel_emb.weight.clone()
    # copyed_rel_depen = [graph_classifier.rel_depen[i].weight for i in range(6)]

    print(f"Device: {params.device}")

    all_auc_roc = []
    auc_roc_mean = 0

    all_auc_pr = []
    auc_pr_mean = 0
    max_label_value = np.array([2, 2])

    triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths,
                                                                             graph_classifier.relation2id)
    new_rel_nums = len(relation2id.keys())
    for r in range(1, params.runs + 1):

        print(ori_rels_num)
        print(new_rel_nums)

        added_rel_emb = nn.Embedding(new_rel_nums, 32, sparse=False).to(device=params.device)
        torch.nn.init.normal_(added_rel_emb.weight)

        for i in range(0, ori_rels_num):
            added_rel_emb.weight[i] = copyed_rel_embed[i]
        #
        graph_classifier.rel_emb.weight.data = added_rel_emb.weight.data

        # influence the random initialization of added_rel_emb
        for i in range(len(graph_classifier.gnn.layers)):
            added_w_comp = nn.Parameter(torch.Tensor(new_rel_nums, 4)).to(device=params.device)
            nn.init.xavier_uniform_(added_w_comp, gain=nn.init.calculate_gain('relu'))
            for j in range(0, ori_rels_num):
                added_w_comp.data[j] = graph_classifier.gnn.layers[i].w_comp.data[j]
                graph_classifier.gnn.layers[i].w_comp.data = added_w_comp.data
            graph_classifier.gnn.layers[i].num_rels = new_rel_nums

        # added_rel_depen = nn.ModuleList([nn.Embedding(new_rel_nums, new_rel_nums) for _ in range(6)]).to(device=params.device)
        # for i in range(6):
        #     torch.nn.init.normal_(added_rel_depen[i].weight)
        #
        # for i in range(6):
        #     for j in range(0, ori_rels_num):
        #         added_rel_depen[i].weight[j, :ori_rels_num] = copyed_rel_depen[i][j]
        # graph_classifier.rel_depen = added_rel_depen

        params.db_path = os.path.join(params.main_dir, f'../data/{params.dataset}/test_subgraphs_{params.model}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

        generate_subgraph_datasets(params, splits=['test'],
                                   saved_relation2id=graph_classifier.relation2id,
                                   max_label_value=max_label_value)

        test = SubgraphDataset(params.db_path, 'test_pos', 'test_neg', params.file_paths, graph_classifier.relation2id,
                               add_traspose_rels=False,
                               num_neg_samples_per_link=params.num_neg_samples_per_link)



        test_evaluator = Evaluator(params, graph_classifier, test)

        result = test_evaluator.eval(save=True)
        print('\nTest Set Performance:' + str(result))
        all_auc_roc.append(result['auc_roc'])
        auc_roc_mean = auc_roc_mean + (result['auc_roc'] - auc_roc_mean) / r

        all_auc_pr.append(result['auc_pr'])
        auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r

    auc_roc_std = np.std(all_auc_roc)
    auc_pr_std = np.std(all_auc_pr)
    avg_auc_roc = np.mean(all_auc_roc)
    avg_auc_pr = np.mean(all_auc_pr)

    print('\nAvg test Set Performance -- mean auc_roc :' + str(avg_auc_roc) + ' std auc_roc: ' + str(auc_roc_std))
    print('\nAvg test Set Performance -- mean auc_pr :' + str(avg_auc_pr) + ' std auc_pr: ' + str(auc_pr_std))

    print(f'auc_pr: {avg_auc_pr: .4f}')

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--expri_name", "-e", type=str, default="default", help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="Toy", help="Dataset string")
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--test_file", "-t", type=str, default="test", help="Name of file containing test triplets")
    parser.add_argument("--runs", type=int, default=5, help="How many runs to perform for mean and std?")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")


    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=100000, help="Set maximum number of links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0, help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1, help="Number of negative examples to sample per positive link")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloading processes")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')
    params = parser.parse_args()
    initialize_experiment(params)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'test': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.test_file))
    }
    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    if torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic = True
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
