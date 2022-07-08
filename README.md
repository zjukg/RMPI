# RMPI

Code and Data for the submission: "Relational Message Passing for Fully Inductive Knowledge Graph Completion".

> In this work, we propose a novel method named RMPI which uses a novel Relational Message Passing network for fully Inductive knowledge graph completion, where the KG is completed with unseen entities and unseen relations newly emerged during testing.
Our proposed RMPI passes messages directly between relations to make full use of the relation patterns for subgraph reasoning with new techniques on graph transformation, graph pruning, relation-aware neighbourhood attention, addressing empty subgraphs, etc., and can utilzes the relation semantics defined in the KG's ontological schema.
Extensive evaluation on multiple benchmarks has shown the effectiveness of RMPI's techniques and its better performance compared with the state-of-the-art methods that support fully inductive KGC as well traditional partially inductive KGC.


### Requirements
The model is developed with PyTorch with environment requirements provided in `requirements.txt`.


### Dataset Illustrations
Each benchmark consists of a training graph and a testing graph.
- In partially inductive KGC, the training graph is denoted as "XXX_vi" in the `data` folder, and the testing graph is denoted as "XXX_vi_ind", where "XXX" means different KGs including WN18RR, FB15k-237 and NELL-995, and "i" means the version index;
- In fully inductive KGC, the training graph is denoted as "XXX_vi", and the testing graph is denoted as "XXX_vi_ind_vj_semi" for testing with semi unseen relations and "XXX_vi_ind_vj_fully" for testing with fully unseen relations, where "j" indicates which version of partially inductive benchmark the testing graph comes from.
For example, for a dataset of NELL-995.v2.v3 in the paper, "nell_v2" is its training graph, while "nell_v2_ind_v3_semi" is its testing graph in the test seting of *testing with semi unseen relations*.

### Model Illustrations
We provide the codes for our RMPI and a baseline of TACT, as well as augment them using ontological schemas with codes contained in the folder `RMPI_S` and `TACT_Base_S`, respectively.


### Basic Training and Testing

To train the model (taking NELL-995.v2 as an example):
```
python RMPI/train.py - d nell_v2 -e nell_v2_base --ablation 0  # for RMPI-base
```


The first thing you need to do is to train the disentangled ontology encoder, using the codes in the folders `OntoEncoder/DOZSL_RD` (for **RD** variants) and `OntoEncoder/DOZSL_AGG` (for **AGG** variants).

**Steps:**
1. Running `run.py` in each method folder to obtain the disentangled concept embeddings;
2. Selecting target **class** or **relation** embeddings from the trained concept embeddings by running `python out_imgc.py` for ZS-IMGC task and `python out_kgc.py` for ZS-KGC task.


#### Entangled ZSL Learner
With the selected class embedding or relation embedding, you can take it to perform downstream ZSL tasks using the generative model or graph propagation model.

The codes for generative model are in folder `ZS_IMGC/models/DOZSL_GAN` and `ZS_KGC/models/DOZSL_GAN` for ZS-IMGC and ZS-KGC tasks, respectively,
for propagation model are in folder `ZS_IMGC/models/DOZSL_GCN` and `ZS_KGC/models/DOZSL_GCN`.

*Note: you can skip the step of training ontology encoder if you just want to use the ontology embedding we learned, the embedding files have already been attached in the corresponding directories*.

#### Baselines
- The baselines for different ZSL methods are in the folders `ZS_IMGC/models` and `ZS_KGC/models` for ZS-IMGC and ZS-KGC tasks, respectively.
- The baselines for different ontology embedding methods are in the folder `OntoEncoder`.
