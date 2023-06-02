# southwest attack
python simulated_averaging_distributed.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 500 \
--part_nets_per_round 30 \
--local_train_period 3 \
--adversarial_local_training_period 3 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-pool \
--attacker_pool_size 100 \
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
--device=cuda:2 \
--number_verifiers 7 \
--clients_per_verifier 30 \
--randomChoose True \
--updateSelection True \
--malicious_verifier reverse  \
> log/latest_reverse_6v_30_3 2>&1

