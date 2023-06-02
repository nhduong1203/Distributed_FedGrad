import numpy as np

import torch
import torch.nn.functional as F
import multiprocessing


# from defense import Con

from models.vgg import get_vgg_model
import pandas as pd

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import *
from helpers import *
from defense_distributed import *
import datasets
import time
import csv
import concurrent.futures

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return output


def get_results_filename(poison_type, attack_method, model_replacement, project_frequency, defense_method, norm_bound, prox_attack, instance="benchmark", fixed_pool=False, model_arch="vgg9"):
    filename = "{}_{}_{}".format(poison_type, model_arch, attack_method)
    if fixed_pool:
        filename += "_fixed_pool" 
    
    if model_replacement:
        filename += "_with_replacement"
    else:
        filename += "_without_replacement"
    
    if attack_method == "pgd":
        filename += "_1_{}".format(project_frequency)
    
    if prox_attack:
        filename += "_prox_attack"

    if defense_method in ("norm-clipping", "norm-clipping-adaptive", "weak-dp"):
        filename += "_{}_m_{}".format(defense_method, norm_bound)
    elif defense_method in ("krum", "multi-krum", "rfa"):
        filename += "_{}".format(defense_method)
               
    filename = f"{instance}_{filename}_acc_results.csv"
    print(f"filename: {filename}")

    return filename


def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="bad"):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    if mode == "bad":
        #pdb.set_trace()
        logger.info("===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "normal":
        logger.info("===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "avg":
        logger.info("===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))

    return norm_diff


def fed_avg_aggregator(net_list, net_freq, device, model="lenet"):
    if model == "lenet":
        net_avg = Net(num_classes=10).to(device)
    elif model in ("vgg9", "vgg11", "vgg13", "vgg16"):
        net_avg = get_vgg_model(model).to(device)
    whole_aggregator = []
    for p_index, p in enumerate(net_list[0].parameters()):
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]
    return net_avg

def weighted_fed_avg_aggregator(net_list, net_freq, c_scores, device, model="lenet"):
    if model == "lenet":
        net_avg = Net(num_classes=10).to(device)
    elif model in ("vgg9", "vgg11", "vgg13", "vgg16"):
        net_avg = get_vgg_model(model).to(device)
    whole_aggregator = []

    # weight_avg = [net_freq[i] * (1/(c_scores[i] + 0.001)) for i in range(len(net_freq))]
    weight_avg = [(1/(c_scores[i] + 0.001)) for i in range(len(net_freq))]
    weight_avg = [weight_avg[i] / sum(weight_avg) for i in range(len(weight_avg))]

    for p_index, p in enumerate(net_list[0].parameters()):
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            print("weight_avg[net_index]", weight_avg[net_index])
            #print("weight_avg[net_index][0]: ", weight_avg[net_index][0])
            params_aggregator = params_aggregator + weight_avg[net_index] * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]
    return net_avg


def estimate_wg(model, device, train_loader, optimizer, epoch, log_interval, criterion):
    logger.info("Prox-attack: Estimating wg_hat")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def train(model, device, train_loader, optimizer, epoch, log_interval, criterion, pgd_attack=False, eps=5e-4, model_original=None,
        proj="l_2", project_frequency=1, adv_optimizer=None, prox_attack=False, wg_hat=None):
    """
        train function for both honest nodes and adversary.
        NOTE: this trains only for one epoch
    """
    model.train()
    # get learning rate
    for param_group in optimizer.param_groups:
        eta = param_group['lr']
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if pgd_attack:
            adv_optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        if prox_attack:
            wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
            model_vec = parameters_to_vector(list(model.parameters()))
            prox_term = torch.norm(wg_hat_vec - model_vec)**2
            loss = loss + prox_term
        
        loss.backward()
        if not pgd_attack:
            optimizer.step()
        else:
            if proj == "l_inf":
                w = list(model.parameters())
                n_layers = len(w)
                # adversarial learning rate
                eta = 0.001
                for i in range(len(w)):
                    # uncomment below line to restrict proj to some layers
                    if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                        w[i].data = w[i].data - eta * w[i].grad.data
                        # projection step
                        m1 = torch.lt(torch.sub(w[i], model_original[i]), -eps)
                        m2 = torch.gt(torch.sub(w[i], model_original[i]), eps)
                        w1 = (model_original[i] - eps) * m1
                        w2 = (model_original[i] + eps) * m2
                        w3 = (w[i]) * (~(m1+m2))
                        wf = w1+w2+w3
                        w[i].data = wf.data
            else:
                # do l2_projection
                adv_optimizer.step()
                w = list(model.parameters())
                w_vec = parameters_to_vector(w)
                model_original_vec = parameters_to_vector(model_original)
                # make sure you project on last iteration otherwise, high LR pushes you really far
                if (batch_idx%project_frequency == 0 or batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > eps):
                    # project back into norm ball
                    w_proj_vec = eps*(w_vec - model_original_vec)/torch.norm(
                            w_vec-model_original_vec) + model_original_vec
                    # plug w_proj back into model
                    vector_to_parameters(w_proj_vec, w)

        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, test_batch_size, criterion, mode="raw-task", dataset="cifar10", poison_type="fashion"):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    if dataset in ("mnist", "emnist"):
        target_class = 7
        if mode == "raw-task":
            classes = [str(i) for i in range(10)]
        elif mode == "targetted-task":
            if poison_type == 'ardis':
                classes = [str(i) for i in range(10)]
            else: 
                classes = ["T-shirt/top", 
                            "Trouser",
                            "Pullover",
                            "Dress",
                            "Coat",
                            "Sandal",
                            "Shirt",
                            "Sneaker",
                            "Bag",
                            "Ankle boot"]
    elif dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # target_class = 2 for greencar, 9 for southwest
        if poison_type in ("howto", "greencar-neo"):
            target_class = 2
        else:
            target_class = 9

    model.eval()
    test_loss = 0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            if poison_type == 'ardis':
                backdoor_index = torch.where(target == target_class)
                target_backdoor = torch.ones_like(target[backdoor_index])
                predicted_backdoor = predicted[backdoor_index]
                backdoor_correct += (predicted_backdoor == target_backdoor).sum().item()
                backdoor_tot = backdoor_index[0].shape[0]


            #for image_index in range(test_batch_size):
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1
    test_loss /= len(test_loader.dataset)

    if mode == "raw-task":
        for i in range(10):
            logger.info('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

            if i == target_class:
                task_acc = 100 * class_correct[i] / class_total[i]

        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        final_acc = 100. * correct / len(test_loader.dataset)

    elif mode == "targetted-task":

        if dataset in ("mnist", "emnist"):
            for i in range(10):
                logger.info('Accuracy of %5s : %.2f %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
            if poison_type == 'ardis':
                # ensure 7 is being classified as 1
                logger.info('Backdoor Accuracy of %.2f : %.2f %%' % (
                     target_class, 100 * backdoor_correct / backdoor_tot))
                final_acc = 100 * backdoor_correct / backdoor_tot
            else:
                # trouser acc
                final_acc = 100 * class_correct[1] / class_total[1]
        
        elif dataset == "cifar10":
            logger.info('#### Targetted Accuracy of %5s : %.2f %%' % (classes[target_class], 100 * class_correct[target_class] / class_total[target_class]))
            final_acc = 100 * class_correct[target_class] / class_total[target_class]
    return final_acc, task_acc

class FederatedLearningTrainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class FixedPoolFederatedLearningTrainer(FederatedLearningTrainer):
    
    def __init__(self, arguments=None, *args, **kwargs):
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacker_pool_size = arguments['attacker_pool_size']
        self.poisoned_emnist_train_loader = arguments['poisoned_emnist_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.attack_method = arguments["attack_method"]
        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.dataset = arguments["dataset"]
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.use_trustworthy = arguments['use_trustworthy']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']
        self.attacker_percent = arguments['attacker_percent']
        self.log_folder = arguments['log_folder']
        self.number_verifiers = arguments['number_verifiers']
        self.clients_per_verifier = arguments['clients_per_verifier']
        self.randomChoose = arguments['randomChoose']
        self.updateSelection = arguments['updateSelection']
        self.malicious_verifier = arguments['malicious_verifier']
        self.reputation_score = [1.0 for _ in range(arguments['num_nets'])] #init reputation score for all clients in the FL systems
        self.local_update_history = [[0.0] for _ in range(arguments['num_nets'])] #theta i,t => keep track of update history of clients
        self.flatten_weights = []
        self.flatten_net_avg = None
        self.instance = arguments['instance']


        logger.info("Posion type! {}".format(self.poison_type))

        if self.attack_method == "pgd":
            self.pgd_attack = True
        else:
            self.pgd_attack = False

        if self.poison_type == 'ardis':
            self.ardis_dataset = datasets.get_ardis_dataset()
            if self.attack_case == 'normal-case':
                self.ardis_dataset.data = self.ardis_dataset.data[66:]
            elif self.attack_case == 'almost-edge-case':
                self.ardis_dataset.data = self.ardis_dataset.data[66:132]
        elif self.poison_type == 'southwest':
            self.ardis_dataset = datasets.get_southwest_dataset(attack_case=self.attack_case)
        else:
            self.ardis_dataset=None

        if arguments["defense_technique"] == "no-defense":
            self._defender = None
        elif arguments["defense_technique"] == "krum":
            num_adv = int(self.attacker_percent*self.part_nets_per_round)
            print(f"num_adv of Krum: {num_adv}")
            self._defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=num_adv)
        elif arguments["defense_technique"] == "multi-krum":
            num_adv = int(self.attacker_percent*self.part_nets_per_round)
            self._defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=num_adv)
        elif arguments["defense_technique"] == "rfa":
            self._defender = RFA()
        elif arguments["defense_technique"] == "fedgrad":
            num_adv = int(self.attacker_percent*self.part_nets_per_round)
            self._defender = FedGrad(total_workers=self.num_nets ,num_workers=self.part_nets_per_round, num_adv=num_adv, num_valid=1, instance=arguments['instance'], use_trustworthy=self.use_trustworthy)
        elif arguments["defense_technique"] == "upper-bound":
            self._defender = UpperBound()
        elif arguments["defense_technique"] == "rlr":
            pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            args_rlr={
                'aggr':'avg',
                'noise':0,
                'clip': 0,
                'server_lr': self.args_lr,
            }
            theta = int(arguments['part_nets_per_round']*self.attacker_percent)+1
            self._defender = RLR(n_params=pytorch_total_params, device=self.device, args=args_rlr, robustLR_threshold=theta)
        elif arguments["defense_technique"] == "flame":
            self._defender = FLAME()
        elif arguments["defense_technique"] == "deepsight":
            self._defender = DeepSight(model_name=self.model, test_batch_size=arguments['test_batch_size'])
            print(f"defender: DeepSight")
        elif arguments["defense_technique"] == "foolsgold":
            pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            self._defender = FoolsGold(num_clients=self.part_nets_per_round, num_classes=10, num_features=pytorch_total_params)
        else:
            NotImplementedError("Unsupported defense method !")
        self.__attacker_pool = np.random.choice(self.num_nets, int(self.num_nets*self.attacker_percent), replace=False)

    def remove(self, attackers, benigns):
        total_dict = {}
        for i in range(len(attackers)):
            atk = attackers[i]
            benign = benigns[i]
            for j in atk:
                if j not in total_dict:
                    total_dict[j] = 0
                total_dict[j] -= 1
            for j in benign:
                if j not in total_dict:
                    total_dict[j] = 0
                total_dict[j] += 1
        
        removed_atk = []
        removed_benign = []
        for i in range(len(attackers)):
            atk = attackers[i]
            benign = benigns[i]
            count = 0
            for j in atk:
                if (total_dict[j] + 1) > 0: 
                    count += 1
            for j in benign:
                if (total_dict[j] - 1) < 0: 
                    count += 1
            if(count < self.clients_per_verifier/3):
                removed_atk.append(atk)
                removed_benign.append(benign)

        return removed_atk, removed_benign

    def run(self, wandb_ins=None):
        main_task_acc = []
        raw_task_acc = []
        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        # additional information tpr_fedgrad, fpr_fedgrad, tnr_fedgrad
        tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 0.0, 0.0, 0.0
        layer1_inf_time, layer2_inf_t, fedgrad_t = 0.0, 0.0, 0.0    
        # let's conduct multi-round training
        self.flatten_net_avg = flatten_model(self.net_avg)
        pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())

        # The number of previous iterations to use FoolsGold on
        memory_size = 0
        delta_memory = np.zeros((self.num_nets, pytorch_total_params, memory_size))
        summed_deltas = np.zeros((self.num_nets, pytorch_total_params))


        # TRAIN
        # trick for set num_net but keep other setting
        num_net = self.num_nets
        for flr in range(1, self.fl_round+1):
            selected_node_indices = np.random.choice(num_net, size=self.part_nets_per_round, replace=False)

            selected_attackers = [idx for idx in selected_node_indices if idx in self.__attacker_pool]
            selected_honest_users = [idx for idx in selected_node_indices if idx not in self.__attacker_pool]
            logger.info("Selected Attackers in FL iteration-{}: {}".format(flr, selected_attackers))
            num_data_points = []
            for sni in selected_node_indices:
                if sni in selected_attackers:
                    num_data_points.append(self.num_dps_poisoned_dataset)
                else:
                    num_data_points.append(len(self.net_dataidx_map[sni]))
            total_num_dps_per_round = sum(num_data_points)
            
            print("num_data_points: ",num_data_points)
            print("total_num_dps_per_round:", total_num_dps_per_round)
            
            net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
            # logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr)) 

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))
            model_original = list(self.net_avg.parameters())
            # super hacky but I'm doing this for the prox-attack
            wg_clone = copy.deepcopy(self.net_avg)
            wg_hat = None
            v0 = torch.nn.utils.parameters_to_vector(model_original)
            wg_norm_list.append(torch.norm(v0).item())

            # start the FL process
            for net_idx, global_user_idx in enumerate(selected_node_indices):
                net  = net_list[net_idx]
                if global_user_idx in selected_attackers:
                    pass
                else:
                    dataidxs = self.net_dataidx_map[global_user_idx]

                    # add p-percent edge-case attack here:
                    if self.attack_case == "edge-case":
                        train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                       self.test_batch_size, dataidxs) # also get the data loader
                    elif self.attack_case in ("normal-case", "almost-edge-case"):
                        train_dl_local, _ = get_dataloader_normal_case(self.dataset, './data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                    else:
                        NotImplementedError("Unsupported attack case ...")
                
                logger.info("@@@@@@@@ Working on client (global-index): {}, which {}-th user in the current round".format(global_user_idx, net_idx))             

                optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # looks like adversary needs same lr to hide with others
                prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
                for param_group in optimizer.param_groups:
                    logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))


                current_adv_norm_diff_list = []
                cnt_attacker = len(selected_attackers)
                if global_user_idx in selected_attackers:

                    if self.prox_attack:
                        # estimate w_hat
                        for inner_epoch in range(1, self.local_training_period+1):
                            estimate_wg(wg_clone, self.device, self.clean_train_loader, prox_optimizer, inner_epoch, log_interval=self.log_interval, criterion=self.criterion)
                        wg_hat = wg_clone

                    for e in range(1, self.adversarial_local_training_period+1):
                        pgd_attack = self.pgd_attack if flr > 1 else False
                       # we always assume net index 0 is adversary
                        if self.defense_technique in ('krum', 'multi-krum'):
                            train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                    pgd_attack=pgd_attack, eps=self.eps*self.args_gamma**(flr-1), model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                    prox_attack=self.prox_attack, wg_hat=wg_hat)
                        elif self.defense_technique == 'kmeans-bases':
                            if flr < 50:
                                train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                    pgd_attack=self.pgd_attack, eps=self.eps*self.args_gamma**(flr-1), model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                    prox_attack=self.prox_attack, wg_hat=wg_hat)
                            else:
                                train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                    pgd_attack=self.pgd_attack, eps=self.eps, model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                    prox_attack=self.prox_attack, wg_hat=wg_hat)
                        else:
                            train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                    pgd_attack=pgd_attack, eps=self.eps, model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                    prox_attack=self.prox_attack, wg_hat=wg_hat)

                    test(net, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
                    test(net, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)

                    # if model_replacement scale models
                    if self.model_replacement:
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))

                        for idx, param in enumerate(net.parameters()):
                            param.data = (param.data - model_original[idx])*(total_num_dps_per_round/self.num_dps_poisoned_dataset) + model_original[idx]
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                    # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                    current_adv_norm_diff_list.append(adv_norm_diff)

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(adv_norm_diff)
                else:
                    for e in range(1, self.local_training_period+1):
                       train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)                
                       # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")
                    
                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(honest_norm_diff)


            #First we update the local updates of each client in this training round
            
            # Time for FoolsGold only
            foolsgold_start_t = time.time()*1000
            if self.defense_technique == "foolsgold":
                delta = np.zeros((self.num_nets, pytorch_total_params))
                if memory_size > 0:
                    for net_idx, global_client_indx in enumerate(selected_node_indices):
                        flatten_local_model = flatten_model(net_list[net_idx])
                        local_update = flatten_local_model - self.flatten_net_avg
                        delta[global_client_indx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[global_client_indx, :]) > 1:
                            delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(delta[global_client_indx, :])

                        delta_memory[global_client_indx, :, flr % memory_size] = delta[global_client_indx, :]
                    # Track the total vector from each individual client
                    summed_deltas = np.sum(delta_memory, axis=2)      
                else:
                    for net_idx, global_client_indx in enumerate(selected_node_indices):
                        flatten_local_model = flatten_model(net_list[net_idx])
                        local_update = flatten_local_model - self.flatten_net_avg
                        local_update = local_update.detach().cpu().numpy()
                        delta[global_client_indx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[global_client_indx, :]) > 1:
                            delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(delta[global_client_indx, :])
                    # Track the total vector from each individual client
                    summed_deltas[selected_node_indices,:] = summed_deltas[selected_node_indices,:] + delta[selected_node_indices,:]
            foolsgold_end_t = time.time()*1000
            foolsgold_t = foolsgold_end_t - foolsgold_start_t

            # MEASURE INFERENCE TIME
            start_t = time.time()*1000
            ### conduct defense here:
            if self.defense_technique == "no-defense":
                pass
            elif self.defense_technique == "krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=num_data_points,
                                                        g_user_indices=selected_node_indices,
                                                        device=self.device)
            elif self.defense_technique == "multi-krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=num_data_points,
                                                        g_user_indices=selected_node_indices,
                                                        device=self.device)
            elif self.defense_technique == "rfa":
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        net_freq=net_freq,
                                                        maxiter=500,
                                                        eps=1e-5,
                                                        ftol=1e-7,
                                                        device=self.device)
            elif self.defense_technique == "flame":
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        net_avg=self.net_avg,
                                                        device=self.device)
            elif self.defense_technique == "foolsgold":
                net_list, net_freq = self._defender.exec(client_models=net_list, delta = delta[selected_node_indices,:], 
                                                         summed_deltas=summed_deltas[selected_node_indices,:], 
                                                         net_avg=self.net_avg, r=flr, device=self.device)
            
            elif self.defense_technique == "deepsight":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        g_t=self.net_avg, 
                                                        num_dps=num_data_points, 
                                                        input_dim=[28,28], 
                                                        g_user_indices=selected_node_indices,
                                                        selected_attackers=selected_attackers,
                                                        model_name=self.model, 
                                                        device=self.device)

            elif self.defense_technique == "fedgrad":

                # control parameter
                weightUpdate = False

                # number clients and verifier
                number_clients_per_verifier = self.clients_per_verifier
                number_verifier = self.number_verifiers

                # setup 
                self._defender.num_workers = number_clients_per_verifier
                self._defender.s = int(self.attacker_percent*number_clients_per_verifier)
                list_node_index = [i for i in range(len(selected_node_indices))]
                verifier_set = random.sample(list(np.arange(num_net)), number_verifier)
                detect_attacker = []
                detect_honest = []
                honest_nets = []
                if (not weightUpdate) or (flr==1):
                    pseudo_avg_net = fed_avg_aggregator(net_list, net_freq, device=self.device, model=self.model)
                else: 
                    pseudo_cs = []
                    for client in selected_node_indices:
                        if (client  in self._defender.accumulate_c_scores):
                            pseudo_cs.append(self._defender.accumulate_c_scores[client])
                        else:
                            pseudo_cs.append(0)
                    pseudo_avg_net = weighted_fed_avg_aggregator(net_list, net_freq, pseudo_cs, device=self.device, model=self.model)

                round_verified = []
                atk_verify = 0
                malicious_verifier = []
                

                for verifier_id in range(len(verifier_set)):
                    # ------------------------------------------------------------------------
                    ## Not Randomly
                    # if not self.randomChoose:
                    #     if(verifier_id % (number_verifier/3) == 0):
                    #         list_node_index = [i for i in range(len(selected_node_indices))]
                    #     this_node_index = random.sample(list_node_index, number_clients_per_verifier)
                    #     list_node_index = [i for i in list_node_index if i not in this_node_index]
                    # else:

                    # list_node_index = [i for i in range(len(selected_node_indices))]
                    this_node_index = random.sample(list_node_index, number_clients_per_verifier) #local
                    round_verified = round_verified + this_node_index
                    round_verified = list(set(round_verified))
                    # ------------------------------------------------------------------------

                    this_node = [selected_node_indices[i] for i in this_node_index] #global
                    this_net_list = [net_list[index] for index in this_node_index] #global
                    this_net_freq = [net_freq[index] for index in this_node_index] #global
                    this_attacker = [i for i in this_node if i in selected_attackers]

                    

                    round_net_list, round_num_dps, pred_g_attacker, pred_g_honest, tpr_fedgrad, fpr_fedgrad, tnr_fedgrad, layer1_inf_time, layer2_inf_t, fedgrad_t = self._defender.exec(client_models=this_net_list,
                                                            num_dps=num_data_points,
                                                            net_freq=this_net_freq,
                                                            net_avg=copy.deepcopy(self.net_avg),
                                                            g_user_indices=this_node, #global
                                                            pseudo_avg_net=pseudo_avg_net,
                                                            round=flr,
                                                            selected_attackers=selected_attackers,
                                                            model_name=self.model,
                                                            device=self.device) 
                    
                    # Reverse attacker's verify result
                    
                    if verifier_set[verifier_id] in self.__attacker_pool:
                        malicious_verifier.append(1)
                        atk_verify += 1
                        if self.malicious_verifier == "reverse": 
                            temp = pred_g_attacker
                            pred_g_attacker = pred_g_honest
                            pred_g_honest = temp

                        elif self.malicious_verifier == "random":
                            s = random.randint(1, len(this_node))
                            pred_g_attacker = np.random.choice(this_node, s, replace=False).tolist()
                            pred_g_honest = []
                            for i in this_node:
                                if i not in pred_g_attacker:
                                    pred_g_honest.append(i)
                    
                        elif self.malicious_verifier == "all_attacker":
                            pred_g_attacker = this_node
                            pred_g_honest = []
                        
                        elif self.malicious_verifier == "all_benign":
                            pred_g_attacker = []
                            pred_g_honest = this_node
                    else:
                        malicious_verifier.append(0)
                    detect_attacker.append(pred_g_attacker)
                    detect_honest.append(pred_g_honest)
                    print(f"verifier: {verifier_id}, true positive: {tpr_fedgrad}, true negative: {tnr_fedgrad}, false positive: {fpr_fedgrad}\n")
                    print(f"predict honest: {pred_g_honest}\n")
                    print(f"predict attacker: {pred_g_attacker}\n")
                    print(f"true attacker: {this_attacker}\n")
                    print("------------------------------------------------------------------------------------")

                
                
                honest_nets = []
                total_num_dps = []
                honest_client = []
                #cs_honest_score = [] # weight average

                voting = True
                if voting:
                    # detect_attacker, detect_honest = self.remove(detect_attacker, detect_honest)
                    print("Log info ...: ")
                    print(detect_attacker)
                    print(detect_honest)
                    print(malicious_verifier)
                    detect_attacker = [element for sublist in detect_attacker for element in sublist]
                    detect_honest = [element for sublist in detect_honest for element in sublist]
                    honest_dict = {index: 0 for index in selected_node_indices}
                    atk_dict = {index: 0 for index in selected_node_indices}
                    for element in detect_attacker:
                        atk_dict[element] += 1
                    for element in detect_honest:
                        honest_dict[element] += 1

                    detect_honest = []
                    detect_attacker = []
                    for element in selected_node_indices:
                        if ((honest_dict[element] + atk_dict[element]) == 0): continue
                        if element in selected_attackers:
                            print ("Attacker: ", honest_dict[element]/(honest_dict[element] + atk_dict[element]))
                        else:
                            print ("Benign: ", honest_dict[element]/(honest_dict[element] + atk_dict[element]))
                        if(honest_dict[element]/(honest_dict[element] + atk_dict[element]) > 0.5):
                            detect_honest.append(element)
                        else:
                            detect_attacker.append(element)
                    print("voting reuslt: ", detect_honest)
                    print("true attacker: ", selected_attackers)
                else:
                    detect_attacker = [element for sublist in detect_attacker for element in sublist]
                    detect_honest = [element for sublist in detect_honest for element in sublist]
                    detect_attacker = list(set(detect_attacker))
                    detect_honest = list(set(detect_honest))
                
                print("detect attacker: ", detect_attacker)
                print("detect honest", detect_honest)

                
                    
                
        
                for idx, client in enumerate(selected_node_indices):
                    if (client not in detect_attacker):
                        if self.updateSelection and (client not in detect_honest):
                            continue
                        honest_nets.append(net_list[idx])
                        total_num_dps.append(num_data_points[idx])
                        honest_client.append(client)
                        # if client in self._defender.accumulate_c_scores:
                        #     cs_honest_score.append(self._defender.accumulate_c_scores[client]) # weight average
                        # else:
                        #     cs_honest_score.append(0) 

                #-----------------------------------------------------------------------------------------------------------------------------------------------------#
                tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 0.0, 0.0, 0.0
                tp_fedgrad_pred = []
                for id_ in selected_attackers:
                    tp_fedgrad_pred.append(1.0 if id_ in detect_attacker else 0.0)
                total_client = len(selected_node_indices)
                fp_fegrad = len(detect_attacker) - sum(tp_fedgrad_pred)
                total_positive = len(selected_attackers)
                total_negative = total_client - total_positive
                tpr_fedgrad = 1.0
                if total_positive > 0.0:
                    tpr_fedgrad = sum(tp_fedgrad_pred)/total_positive
                fpr_fedgrad = 0
                if total_negative > 0.0:
                    fpr_fedgrad = fp_fegrad/total_negative
                tnr_fedgrad = 1.0 - fpr_fedgrad
                #-----------------------------------------------------------------------------------------------------------------------------------------------------#

                #-----------------------------------------------------------------------------------------------------------------------------------------------------#
                print(f"Flr: {flr}, number verifier: {number_verifier}, clients/verifier: {number_clients_per_verifier}, attacked verifier: {atk_verify}\n")
                print(f"Flr: {flr}, true positive: {tpr_fedgrad}, true negative: {tnr_fedgrad}, false positive: {fpr_fedgrad}\n")
                print(f"Flr: {flr}, honest client after all verify: {honest_client}")
                print(f"Flr: {flr}, attackers: {selected_attackers}")
                not_verified = len(selected_node_indices) - len(round_verified)
                print(f'Flr: {flr}, number not verify: {not_verified}')

                self._defender.update_trustworthy(detect_attacker, detect_honest)
                self._defender.update()


                if not weightUpdate:
                    if len(honest_nets) != 0:
                        freq_nets = [snd/sum(total_num_dps) for snd in total_num_dps]
                        self.net_avg = fed_avg_aggregator(honest_nets, freq_nets, device=self.device, model=self.model)
                else:
                    if len(honest_nets) != 0:
                        freq_nets = [snd/sum(total_num_dps) for snd in total_num_dps]
                        self.net_avg = weighted_fed_avg_aggregator(honest_nets, freq_nets, cs_honest_score, device=self.device, model=self.model)


            
            elif self.defense_technique == 'rlr':
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        num_dps=num_data_points,
                                                        global_model=self.net_avg)
        
            else:
                NotImplementedError("Unsupported defense method !")
            
            end_t = time.time()*1000
            ref_time = end_t - start_t # For whole defenses
            if self.defense_technique == 'foolsgold':
                ref_time += foolsgold_t
            
            # after local training periods
            # First we update the local updates of each client in this training round
            if self.defense_technique != "fedgrad":
                self.net_avg = fed_avg_aggregator(net_list, net_freq, device=self.device, model=self.model)
            self.flatten_net_avg = flatten_model(self.net_avg)
            end_full_t = time.time()*1000
            full_inf_t = end_full_t - start_t

            # Logging here 
            # logging_filename = f'logging/backdoor_metadata.csv' #metadata file
            # logging_items = get_logging_items(net_list, selected_node_indices, prev_avg, self.net_avg, selected_attackers, flr)
            # with open(logging_filename, 'a+') as lf:
            #     write = csv.writer(lf)
            #     write.writerows(logging_items)

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.net_avg, epoch=0, fl_round=flr, mode="avg")
            
            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))
            overall_acc, raw_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            backdoor_acc, _ = test(self.net_avg, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
 
            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            raw_task_acc.append(raw_acc)
            backdoor_task_acc.append(backdoor_acc)
            
            adv_norm_diff = 1.0*sum(current_adv_norm_diff_list)/len(current_adv_norm_diff_list) if len(current_adv_norm_diff_list) else 0

            adv_norm_diff_list.append(adv_norm_diff)
            if(wandb_ins):
                wandb_logging = {'fl_iter': flr, 
                            'main_task_acc': overall_acc, 
                            'backdoor_acc': backdoor_acc, 
                            'raw_task_acc':raw_acc, 
                            'adv_norm_diff': adv_norm_diff, 
                            'wg_norm': torch.norm(v0).item(),
                            'cnt_attackers': cnt_attacker,
                            'ref_time': ref_time,
                            'full_inf_t': full_inf_t,
                            }
                additional_logging = {
                    'tpr_fedgrad': tpr_fedgrad, 
                    'fpr_fedgrad': fpr_fedgrad, 
                    'tnr_fedgrad': tnr_fedgrad,
                    'layer1_inf_time': layer1_inf_time, 
                    'layer2_inf_t': layer2_inf_t,
                    'fedgrad_t':fedgrad_t,
                }
                wandb_ins.log({"general": wandb_logging, "additional": additional_logging})
                # wandb_ins.log({"additional": additional_logging})
                
            
        df = pd.DataFrame({'fl_iter': fl_iter_list, 
                            'main_task_acc': main_task_acc, 
                            'backdoor_acc': backdoor_task_acc, 
                            'raw_task_acc':raw_task_acc, 
                            'adv_norm_diff': adv_norm_diff_list, 
                            'wg_norm': wg_norm_list
                            })
       
        if self.poison_type == 'ardis':
            # add a row showing initial accuracies
            df1 = pd.DataFrame({'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]})
            df = pd.concat([df1, df])

        results_filename = get_results_filename(self.poison_type, self.attack_method, self.model_replacement, self.project_frequency,
                self.defense_technique, self.norm_bound, self.prox_attack, fixed_pool=True, model_arch=self.model, instance=self.instance)
        df.to_csv(results_filename, index=False)

        logger.info("Wrote accuracy results to: {}".format(results_filename))

