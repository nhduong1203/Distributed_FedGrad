python simulated_averaging_distributed.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 3383 \
--fl_round 500 \
--rand_seed 100 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset emnist \
--model lenet \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
--defense_method fedgrad \
--attack_method blackbox \
--wandb_group TestingGroup \
--instance emnist_test_pdr_40 \
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
--clients_per_verifier 6 \
--randomChoose True \
--updateSelection True \
--malicious_verifier normal \
--device=cuda:1 \
> log/emnist_test_pdr_50.txt 2>&1
# --number_verifiers 15 \
# --clients_per_verifier 6 \
# --randomChoose True \
# --updateSelection True \
# --malicious_verifier normal \


# python simulated_averaging_distributed.py --fraction 0.15 \
# --lr 0.02 \
# --gamma 0.998 \
# --num_nets 3383 \
# --fl_round 500 \
# --part_nets_per_round 30 \
# --local_train_period 3 \
# --adversarial_local_training_period 3 \
# --dataset emnist \
# --model lenet \
# --fl_mode fixed-pool \
# --attacker_pool_size 100 \
# --defense_method fedgrad \
# --attack_method blackbox \
# --wandb_group TestingGroup \
# --instance test_mnist \
# --attack_case edge-case \
# --model_replacement False \
# --project_frequency 1 \
# --stddev 0.025 \
# --eps 2 \
# --adv_lr 0.02 \
# --prox_attack False \
# --poison_type ardis \
# --norm_bound 2 \
# --attacker_percent 0.25 \
# --pdr 0.5 \
# --degree_nonIID 0.5 \
# --use_trustworthy True \
# --number_verifiers 15 \
# --clients_per_verifier 7 \
# --randomChoose True \
# --updateSelection True \
# --malicious_verifier normal \
# --device=cuda:0 \
# > log/test_mnist 2>&1




# southwest attack
# python simulated_averaging_distributed.py --fraction 0.1 \
# --lr 0.02 \
# --gamma 0.998 \
# --num_nets 200 \
# --fl_round 500 \
# --part_nets_per_round 30 \
# --local_train_period 3 \
# --adversarial_local_training_period 3 \
# --dataset cifar10 \
# --model vgg9 \
# --fl_mode fixed-pool \
# --attacker_pool_size 100 \
# --wandb_group TestingGroup \
# --instance my_server_test \
# --defense_method fedgrad \
# --attack_method blackbox \
# --attack_case edge-case \
# --model_replacement False \
# --project_frequency 10 \
# --stddev 0.025 \
# --eps 2 \
# --adv_lr 0.02 \
# --prox_attack False \
# --poison_type southwest \
# --norm_bound 2 \
# --attacker_percent 0.25 \
# --pdr 0.33 \
# --degree_nonIID 0.5 \
# --use_trustworthy True \
# --device=cuda:3 \
# --number_verifiers 23 \
# --clients_per_verifier 5 \
# --randomChoose True \
# --updateSelection True \
# --malicious_verifier random \
# > log/23V_5L_0.25_random_selectVerifier_noAccumulate 2>&1

