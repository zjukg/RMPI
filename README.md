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
We provide the codes for our RMPI and a baseline of TACT, and augment them using ontological schemas with the codes contained in the folder `RMPI_S` and `TACT_Base_S`, respectively.


### Basic Training and Testing of RMPI and its variants

To train the model (taking NELL-995.v2 as an example):
```
# for RMPI-base
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_base --ablation 0
# for RMPI-NE with summation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE --ablation 1
#for RMPI-NE with concatenation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE_conc --ablation 1 --conc
# for RMPI-TA
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_TA --ablation 0 --target2nei_atten
# for RMPI-NE-TA with summation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE_TA --ablation 1  --target2nei_atten
# for RMPI-NE-TA with concatenation-based fusion function
python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_NE_conc_TA --ablation 1 --conc --target2nei_atten
```
- `-e` represents the saved model names, depending on the applied variants.
- `-d` represents the target benchmarks.

To test the model (taking RMPI-base and *testing with semi unseen relation* as an example):
- Fully inductive case & Triple classification
```
python RMPI/test_auc_F.py -d nell_v2_ind_v3_semi -e nell_v2_RMPI_base --ablation 0
```
- Fully inductive case & Entity Prediction
```
python RMPI/test_ranking_F.py -d nell_v2_ind_v3_semi -e nell_v2_RMPI_base --ablation 0
```
- Partially inductive case & Entity Prediction
```
python RMPI/test_ranking_P.py -d nell_v2_ind -e nell_v2_RMPI_base --ablation 0
```

`RMPI_S` is trained and tested in a similar way.


### Basic Training and Testing of TACT and TACT-base

To train the model (taking NELL-995.v2 as an example):
```
# for TACT-base
python TACT/train.py -d nell_v2 -e nell_v2_TACT_base --ablation 3
# for TACT full model
python TACT/train.py -d nell_v2 -e nell_v2_TACT --ablation 0
```

To test the model (taking "TACT-base & Fully inductive case & *testing with semi unseen relation* & Triple classification" as an example):
```
python TACT/test_auc_F.py -d nell_v2_ind_v3_semi -e nell_v2_TACT_base --ablation 3
```

`TACT_S` is trained and tested in a similar way.


### Pre-training of Ontological schema
- The pre-training is implemented by running the open codes of TransE provided in [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) on the schema graph.
- The schema graph (`Schema-NELL.csv`) and pre-trained embeddings have been attached in the folder `data/external_rel_embeds`. Thanks for the resource from [KZSL](https://github.com/China-UK-ZSL/Resources_for_KZSL).

## Some Results for Supplementing Main Paper
1. Entity Prediction on WN18RR in the partially inductive KGC.


<table>
    <tr>  
        <th rowspan="2">Method</th><th colspan="4">WN18RR.v1</th><th colspan="4">WN18RR.v2</th>
    </tr>
    <tr>  
       <th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th>
    </tr>
    <tr>
        <td>TACT-base</td><td>80.62</td><td>77.93</td><td>82.45</td><td>82.45</td><td>78.11</td><td>76.76</td><td>78.68</td><td>78.68</td>
    </tr>
    <tr>
        <td>TACT</td><td>79.56</td><td>76.33</td><td>82.45</td><td>82.45</td><td>78.55</td><td>76.87</td><td>78.68</td><td>78.68</td>
    </tr>
    <tr>
        <td>RMPI-base</td><td>79.69</td><td>76.60</td><td>82.18</td><td>82.45</td><td>78.02</td><td>76.53</td><td>78.68</td><td>78.68</td>
    </tr>
    <tr>
        <td>RMPI-NE</td><td>81.58</td><td>75.53</td><td>88.03</td><td>89.63</td><td>81.07</td><td>78.68</td><td>82.31</td><td>83.22</td>
    </tr>
    <tr>
        <td>RMPI-TA</td><td>69.73</td><td>58.51</td><td>82.45</td><td>82.45</td><td>78.13</td><td>76.76</td><td>78.68</td><td>78.68</td>
    </tr>
    <tr>
        <td>RMPI-NE-TA</td><td>81.74</td><td>77.13</td><td>86.44</td><td>87.77</td><td>81.34</td><td>79.82</td><td>81.97</td><td>82.43</td>
    </tr>
</table>

<table>
    <tr>  
        <th rowspan="2">Method</th><th colspan="4">WN18RR.v3</th><th colspan="4">WN18RR.v4</th>
    </tr>
    <tr>  
       <th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th>
    </tr>
    <tr>
        <td>TACT-base</td><td>54.42</td><td> 50.58 </td><td>57.19 </td><td>58.84 </td><td>73.35 </td><td>72.29 </td><td>73.34 </td><td>73.34 </td>
    </tr>
    <tr>
        <td>TACT</td><td> 54.21</td><td> 50.00</td><td> 57.19</td><td> 58.60</td><td> 73.28</td><td> 72.04</td><td> 73.41</td><td> 73.41</td>
    </tr>
    <tr>
        <td>RMPI-base</td><td>55.93 </td><td>52.56 </td><td>58.18 </td><td>58.68 </td><td>73.43 </td><td>72.32 </td><td>73.41 </td><td>73.41 </td>
    </tr>
    <tr>
        <td>RMPI-NE</td><td> 64.85</td><td>60.17 </td><td>68.43 </td><td>70.33 </td><td>77.14 </td><td>74.95 </td><td> 78.13</td><td>79.81 </td>
    </tr>
    <tr>
        <td>RMPI-TA</td><td>56.23 </td><td>52.81 </td><td>57.93 </td><td>58.84 </td><td> 73.68</td><td>72.15 </td><td>73.41 </td><td>73.41 </td>
    </tr>
    <tr>
        <td>RMPI-NE-TA</td><td>65.62</td><td>60.08 </td><td>70.08</td><td>73.14 </td><td>77.68 </td><td>74.84 </td><td>79.50 </td><td>81.42 </td>
    </tr>
</table>


2. Entity Prediction on NELL-995.v1 in the partially inductive KGC.

<table>
    <tr>  
        <th rowspan="2">Method</th><th colspan="4">WN18RR.v3</th>
    </tr>
    <tr>
       <th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th>
    </tr>
    <tr>
        <td>TACT-base</td><td>49.76</td><td>44.00 </td><td>53.00 </td><td>56.50</td>
    </tr>
    <tr>
        <td>TACT</td><td>47.68 </td><td>43.50 </td><td>48.00 </td><td>51.50 </td>
    </tr>
    <tr>
        <td>RMPI-base</td><td>53.43</td><td>48.00</td><td>57.00 </td><td>59.50</td>
    </tr>
    <tr>
        <td>RMPI-NE</td><td> 54.05</td><td>49.50 </td><td>55.00 </td><td>60.50 </td>
    </tr>
    <tr>
        <td>RMPI-TA</td><td> 48.97</td><td>44.00 </td><td>52.50 </td><td> 53.00</td>
    </tr>
    <tr>
        <td>RMPI-NE-TA</td<td>54.24 </td><td>50.00 </td><td>55.50 </td><td>60.50 </td>
    </tr>
</table>

