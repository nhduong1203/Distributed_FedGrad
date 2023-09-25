
### Depdendencies (tentative)
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1 -->
### Depdendencies (tentative)
---
The requirements are listed on requirements.txt file.

### Data Preparation
---
1. For Southwest Airline (for CIFAR-10) and traditional Cretan costumes (for ImageNet) edge-case example, most of the collected edge-case datasets are available in `./saved_datasets`. 
2. To get the `Ardis` dataset, the edge-case datasets can be generated via running `get_ardis_data.sh` and then `generating_poisoned_DA.py`.

### Running Experients:
---
The main script is `./simulated_averaging.py`, to launch the jobs, we provide a script `./run_simulated_averaging.sh`. And we provide a detailed description on the arguments.


| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `fraction` | Only used for EMNIST, varying the fraction of poisioned data points in attacker's poisoned dataset. |
| `lr` | Inital learning rate that will be used for local training process. |
| `batch-size` | Batch size for the optimizers e.g. SGD or Adam. |
| `dataset`      | Dataset to use. |
| `model`      | Model to use. |
| `gamma` | the factor of learning rate decay, i.e. the effective learning rate is `lr*gamma^t`. |
| `batch-size` | Batch size for the optimizers e.g. SGD or Adam. |
| `num_nets` | The total number of available users e.g. 3383 for EMNIST and 200 for CIFAR-10. |
| `fl_round` | maximum number of FL rounds for the code to run. |
| `part_nets_per_round` | Number of active users that are sampled per FL round to participate. |
| `local_train_period` | Number of local training epochs that the honest users can run. |
| `adversarial_local_training_period`  | Number of local training epochs that the attacker(s) can run. |
| `fl_mode`    | `fixed-freq` or `fixed-pool` for fixed frequency and fixed pool attacking settings.  |
| `defense_method`    | Defense method over the data center end.   |
| `stddev` | Standard deviation of the noise added for weak DP defense. |
| `norm_bound` | Norm bound for the norm difference clipping defense. |
| `attack_method` | Attacking schemes used for attacker and either be `blackbox` or `PGD`. |
| `attack_case` | Wether or not to conduct edge-case attack, can be `edge-case`, `normal-case` or `almost-edge-case`. |
| `model_replacement` | Used when `attack_method=PGD` to control if the attack is PGD with replacement or without replacement. |
| `project_frequency` | How frequent (in how many iterations) to project to the l2 norm ball in PGD attack. |
| `eps` | Radius the l2 norm ball in PGD attack. |
| `adv_lr` | Learning rate of the attacker when conducting PGD attack. |
| `poison_type` | Specify the backdoor for each dataset using `southwest` for CIFAR-10 and `ardis` for EMNIST. |
| `device` | Specify the hardware to run the experiment. |
| `attacker_percent` | The percentage of attackers per all clients. |
| `use_trustworthy` | Turn on/off trustworthy filtering layer in FedGrad. |
| `degree_nonIID` | The degree_nonIID of data distribution between clients. |
| `pdr` | The poisoned data rate inside training data of a compromised client. |


#### Sample command
Blackbox attack on Southwest Airline exmaple over CIFAR-19 dataset where there is no defense on the data center. The attacker participate in the fixed-pool manner.
```
python simulated_averaging.py \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-pool \
--defense_method fedgrad \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--attacker_percent 0.25 \
--pdr 0.33 \
--degree_nonIID 0.5 \
--use_trustworthy True \
--device=cuda:1
``` 

### Acknowledgement
We would like to send a big Acknowledgement to the authors of "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning".\
[OOD_Federated_Learning](https://github.com/ksreenivasan/OOD_Federated_Learning)
<!-- ### Experiment guide (by Dung):
---
0. Set up appropriate environment with all packgages listed on file `requirements.txt`
1. Before starting, run: `get_ardis_data.sh` then -> `python generating_poisoned_DA.py`, to generate poisoning data
2. Folder containing all bash files to run experiments:
    ```
    bash-experiment-all
        --scenario1.1 
        --scenario1.2
    ``` -->
