# RMPI

Code and Data for the submission: "Relational Message Passing for Fully Inductive Knowledge Graph Completion".

> In this work, we propose a novel method named RMPI which uses a novel *R*elational *M*essage *P*assing network for fully *I*nductive knowledge graph completion, where the KG is completed with unseen entities and unseen relations newly emerged during testing.
<br>Our proposed RMPI passes messages directly between relations to make full use of the relation patterns for subgraph reasoning with new techniques on graph transformation, graph pruning, relation-aware neighbourhood attention, addressing empty subgraphs, etc., and can utilize the relation semantics defined in the KG's ontological schema.
<br>Extensive evaluation on multiple benchmarks has shown the effectiveness of RMPI's techniques and its better performance compared with the state-of-the-art methods that support fully inductive KGC as well traditional partially inductive KGC.


### Requirements
The model is developed using PyTorch with environment requirements provided in `requirements.txt`.


### Dataset Illustrations
Each benchmark consists of a training graph and a testing graph.
- In partially inductive KGC, the training graph is denoted as "XXX_vi", and the testing graph is denoted as "XXX_vi_ind", where "XXX" means different KGs including WN18RR, FB15k-237 and NELL-995, and "i" means the version index;
- In fully inductive KGC, the training graph is denoted as "XXX_vi", and the testing graph is denoted as "XXX_vi_ind_vj_semi" for testing with semi unseen relations and "XXX_vi_ind_vj_fully" for testing with fully unseen relations, where "j" indicates which version of partially inductive benchmark the testing graph comes from.


For example, for a dataset of NELL-995.v2.v3 in the paper, "nell_v2" is its training graph, while "nell_v2_ind_v3_semi" is its testing graph in the test seting of *testing with semi unseen relations*.

### Model Illustrations
We provide the codes for our RMPI and a baseline of TACT, and augmented them using ontological schemas with codes contained in the folder `RMPI_S` and `TACT_Base_S`, respectively.


### Basic Training and Testing of RMPI and its variants

To train the model (taking NELL-995.v2 as an example):
```
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_base --ablation 0  #for RMPI-base
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE --ablation 1  #for RMPI-NE with summation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE_conc --ablation 1 --conc  #for RMPI-NE with concatenation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_TA --ablation 0 --target2nei_atten  #for RMPI-TA
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE_TA --ablation 1  --target2nei_atten #for RMPI-NE-TA with summation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE_conc_TA --ablation 1 --conc --target2nei_atten #for RMPI-NE-TA with concatenation-based fusion function
```
- `-e` represents the saved model names, depending on the applied variants.
- `-d` represents the target benchmarks.

To test the model (taking RMPI-base and *testing with semi unseen relation* as an example):
- fully inductive case & Triple classification
```
python RMPI/test_auc_F.py -d nell_v2_ind_v3_semi -e nell_v2_RMPI_base --ablation 0
```
- fully inductive case & Entity Prediction
```
python RMPI/test_ranking_F.py -d nell_v2_ind_v3_semi -e nell_v2_RMPI_base --ablation 0
```
- partially inductive case & Entity Prediction
```
python RMPI/test_ranking_P.py -d nell_v2_ind -e nell_v2_RMPI_base --ablation 0
```

**`RMPI_S` is trained and tested in a similar way.**


### Basic Training and Testing of TACT and TACT-base

To train the model (taking NELL-995.v2 as an example):
```
python TACT/train.py -d nell_v2 -e nell_v2_TACT_base --ablation 3  #for TACT-base
python TACT/train.py -d nell_v2 -e nell_v2_TACT --ablation 0  #for TACT full model
```

To test the model (taking "TACT-base & *testing with semi unseen relation* & fully inductive case & Triple classification" as an example):
```
python TACT/test_auc_F.py -d nell_v2_ind_v3_semi -e nell_v2_TACT_base --ablation 3
```

**`TACT_S` is trained and tested in a similar way.**


### Pre-training of Ontological schema
The pre-training is implemented by running the open codes of TransE provided in [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) on the schema graph.
The schema graph (`Schema-NELL.csv`) and pre-trained embeddings have been attached in the folder `data/external_rel_embeds'. Thanks for the resource from [KZSL](https://github.com/China-UK-ZSL/Resources_for_KZSL).