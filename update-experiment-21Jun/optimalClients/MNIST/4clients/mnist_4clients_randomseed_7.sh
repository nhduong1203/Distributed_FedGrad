#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=54:00:00
#$ -o /home/aaa10078nj/Federated_Learning//HaiDuong_DistFedGrad/logs/optimalClients/MNIST/4clients/$JOB_NAME_$JOB_ID.log
#$ -j y
​
source /etc/profile.d/modules.sh
#module load gcc/11.2.0
#Old gcc. Newest support is 12.2.0. See module avail
LD_LIBRARY_PATH=/apps/centos7/gcc/11.2.0/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/gcc/11.2.0/bin:${PATH}
#module load openmpi/4.1.3
#Old mpi. Use intel mpi instead
LD_LIBRARY_PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/bin:${PATH}
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
#module load python/3.10/3.10.4
#Old python. Newest support is 10.3.10.10. See module avail
LD_LIBRARY_PATH=/apps/centos7/python/3.10.4/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/python/3.10.4/bin:${PATH}
​
source ~/venv/pytorch1.11+horovod/bin/activate
python --version
LOG_DIR="/home/aaa10078nj/Federated_Learning/HaiDuong_DistFedGrad/logs/optimalClients/MNIST/4clients/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
​
cp -rp /home/aaa10078nj/Federated_Learning/HaiDuong_DistFedGrad/Distributed_FedGrad $SGE_LOCALDIR/$JOB_ID/
cd $SGE_LOCALDIR/$JOB_ID
#cd Distributed_FedGrad
​
# southwest attack
python simulated_averaging_distributed.py --fraction 0.15 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 3383 \
--fl_round 1000 \
--rand_seed 600 \
--part_nets_per_round 30 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset emnist \
--model lenet \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method fedgrad \
--attack_method blackbox \
--wandb_group newOptimalClientsGroup \
--instance emnist_15Verifiers_4Clients \
--attack_case edge-case \
--model_replacement False \
--project_frequency 1 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type ardis \
--norm_bound 2 \
--attacker_percent 0.25 \
--pdr 0.5 \
--degree_nonIID 0.5 \
--use_trustworthy True \
--number_verifiers 15 \
--clients_per_verifier 4 \
--randomChoose True \
--updateSelection True \
--malicious_verifier normal \
--log_folder ${LOG_DIR} \
--device=cuda
#> log/exp1 2>&1