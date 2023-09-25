class FedGrad(Defense):
    """
    FedGrad by DungNT
    """
    def __init__(self, total_workers, num_workers, num_adv, num_valid = 1, instance="benchmark", use_trustworthy=False, *args, **kwargs):
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv
        self.instance = instance
        self.choosing_frequencies = {}
        self.accumulate_c_scores = {}
        self.pseudo_accumulate_c_scores =  {}
        self.pseudo_choosing_frequencies = {}
        self.list_ac_sc = {}
        self.use_trustworthy = use_trustworthy
        self.pairwise_w = np.zeros((total_workers+1, total_workers+1))
        self.pairwise_b = np.zeros((total_workers+1, total_workers+1))
        self.eta = 0.5 # this parameter could be changed
        self.switch_round = 50 # this parameter could be changed
        self.trustworthy_threshold = 0.75
        self.lambda_1 = 0.25
        self.lambda_2 = 1.0
        
        self.pairwise_choosing_frequencies = np.zeros((total_workers, total_workers))
        self.trustworthy_scores = [[0.5] for _ in range(total_workers+1)]

    def update(self):
        self.choosing_frequencies = self.pseudo_choosing_frequencies.copy()
        for key, value in self.list_ac_sc.items():
            if len(value) != 0:
                self.accumulate_c_scores[key] = sum(value)/len(value)
            self.list_ac_sc[key] = []

    def update_trustworthy(self, detect_attacker, detect_honest):
        for client in detect_attacker:
            self.trustworthy_scores[client].append(self.lambda_1)

        for client in detect_honest:
            self.trustworthy_scores[client].append(self.lambda_2)


    def exec(self, client_models, num_dps, net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, model_name, device, list_client = None, weight_avg = False, *args, **kwargs):
        start_fedgrad_t = time.time()*1000

        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        neighbor_distances = []
        
        #-------------------------------------- SOFT FILTER -----------------------------------------------------------
        layer1_start_t = time.time()*1000
        bias_list, _, _, _, weight_update, glob_update, _ = extract_classifier_layer(client_models, pseudo_avg_net, net_avg, model_name)
        total_client = len(g_user_indices)
        
        raw_c_scores = self.get_compromising_scores(glob_update, weight_update)
        c_scores = [] 

        for idx, cli in enumerate(g_user_indices):
            self.pseudo_choosing_frequencies[cli] = self.choosing_frequencies.get(cli, 0) + 1
            self.pseudo_accumulate_c_scores[cli] = ((self.pseudo_choosing_frequencies[cli] - 1) / self.pseudo_choosing_frequencies[cli]) * self.accumulate_c_scores.get(cli, 0) + (1 / self.pseudo_choosing_frequencies[cli]) *  raw_c_scores[idx]
            
            if cli not in self.list_ac_sc:
                self.list_ac_sc[cli] = []
            self.list_ac_sc[cli].append(self.pseudo_accumulate_c_scores[cli][0]) # append after update
            c_scores.append(self.pseudo_accumulate_c_scores[cli])

        c_scores = np.array(c_scores)
        epsilon_1 = min(self.eta, np.median(c_scores))
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        
        suspicious_idxs_1 = [ind_ for ind_ in range(total_client) if c_scores[ind_] > epsilon_1] #local
        global_suspicious_idxs_1 = [g_user_indices[index] for index in suspicious_idxs_1] #global
        layer1_end_t = time.time()*1000
        layer1_inf_time = layer1_end_t-layer1_start_t
        print("epsilon_1: ",epsilon_1)
        print("[Soft-filter] predicted suspicious set is: ", global_suspicious_idxs_1)
        print(f"Total computation time of the 1st layer is: {layer1_inf_time}")
        
        # ----------------------------------- HARD FILTER --------------------------------------------------------------
        layer2_start_t = time.time()*1000
        round_pw_bias = np.zeros((total_client, total_client))
        round_pw_weight = np.zeros((total_client, total_client))
        
        sum_diff_by_label, _ = calculate_sum_grad_diff(meta_data = weight_update, num_w = weight_update[0].shape[-1], glob_update=glob_update)
        norm_bias_list = normalize(bias_list, axis=1)
        norm_grad_diff_list = normalize(sum_diff_by_label, axis=1)
        
        # UPDATE CUMULATIVE COSINE SIMILARITY 
        for i, g_i in enumerate(g_user_indices):
            for j, g_j in enumerate(g_user_indices):
                bias_p_i = norm_bias_list[i]
                bias_p_j = norm_bias_list[j]
                cs_1 = np.dot(bias_p_i, bias_p_j)/(np.linalg.norm(bias_p_i)*np.linalg.norm(bias_p_j))
                round_pw_bias[i][j] = cs_1.flatten()

                w_p_i = norm_grad_diff_list[i]
                w_p_j = norm_grad_diff_list[j]
                cs_2 = np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
                round_pw_weight[i][j] = cs_2.flatten()
       
        # compute closeness scores 
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2))
            neighbor_distances.append(distance)

        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        trusted_index = scores.index(min(scores)) # ==> trusted client is the client whose smallest closeness score.

        scaler = MinMaxScaler()
        round_pw_bias = scaler.fit_transform(round_pw_bias)
        round_pw_weight = scaler.fit_transform(round_pw_weight)
                  
        # From now on, trusted_model contains the index base model treated as valid user.
        suspicious_idxs_2, saved_pairwise_sim, layer2_inf_t = [], [], 0.0
        
        # NOW CHECK FOR SWITCH ROUND
        # TODO: find dynamic threshold
        # STILL PERFORM HARD-FILTER to save the historical information about colluding property.

        # use round information instead
        saved_pairwise_sim = np.hstack((round_pw_weight, round_pw_bias))
        kmeans = KMeans(n_clusters = 2)
        pred_labels = kmeans.fit_predict(saved_pairwise_sim)
        trusted_cluster_idx = pred_labels[trusted_index] # assign cluster containing trusted client as benign cluster
        malicious_cluster_idx = 0 if trusted_cluster_idx == 1 else 1
        suspicious_idxs_2 = np.argwhere(np.asarray(pred_labels) == malicious_cluster_idx).flatten() # local
        
        layer2_end_t = time.time()*1000
        layer2_inf_t = layer2_end_t-layer2_start_t

        global_suspicious_idxs_2 = [g_user_indices[index] for index in suspicious_idxs_2]
        print("[Hard-filter] predicted suspicious set is: ", global_suspicious_idxs_2)
        print(f"Total computation time of the 2nd layer is: {layer2_inf_t}")
        # ----------------------------------- HARD FILTER --------------------------------------------------------------
        final_suspicious_idxs = suspicious_idxs_1 
        pseudo_final_suspicious_idxs = np.union1d(suspicious_idxs_2, suspicious_idxs_1).flatten().astype(int)
        if round >= self.switch_round:
            final_suspicious_idxs = pseudo_final_suspicious_idxs
            
        # STARTING USING TRUSTWORTHY SCORES
        filtered_suspicious_idxs = list(final_suspicious_idxs.copy())
        if round >= self.switch_round:
            filtered_suspicious_idxs = [idx for idx in final_suspicious_idxs if np.average(self.trustworthy_scores[g_user_indices[idx]]) < self.trustworthy_threshold]
        if not filtered_suspicious_idxs:
            filtered_suspicious_idxs = suspicious_idxs_1     
                 
        if self.use_trustworthy: # used for ablation study
            final_suspicious_idxs = filtered_suspicious_idxs
        global_final_suspicious_idxs= [g_user_indices[index] for index in final_suspicious_idxs]
        print(f"[Final-result] predicted suspicious set is: {global_final_suspicious_idxs}")   
        
        #GET ADDITIONAL INFORMATION of TPR and FPR, TNR
        tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 0.0, 0.0, 0.0
        tp_fedgrad_pred = []
        for id_ in participated_attackers:
            tp_fedgrad_pred.append(1.0 if id_ in final_suspicious_idxs else 0.0)
        fp_fegrad = len(final_suspicious_idxs) - sum(tp_fedgrad_pred)
        
        # Calculate true positive rate (TPR = TP/(TP+FN))
        total_positive = len(participated_attackers)
        total_negative = total_client - total_positive
        tpr_fedgrad = 1.0
        if total_positive > 0.0:
            tpr_fedgrad = sum(tp_fedgrad_pred)/total_positive
        fpr_fedgrad = 0
        if total_negative > 0.0:
            fpr_fedgrad = fp_fegrad/total_negative
        tnr_fedgrad = 1.0 - fpr_fedgrad
        
        end_fedgrad_t = time.time()*1000
        fedgrad_t = end_fedgrad_t - start_fedgrad_t # finish calculating the computation time of FedGrad
        
        neo_net_list, selected_num_dps = [], []
        
        pred_g_attacker = [g_user_indices[i] for i in final_suspicious_idxs]
        pred_g_honest = [user_index for user_index in g_user_indices if user_index not in pred_g_attacker]

        return neo_net_list, selected_num_dps, pred_g_attacker, pred_g_honest, tpr_fedgrad, fpr_fedgrad, tnr_fedgrad, layer1_inf_time, layer2_inf_t, fedgrad_t

    def get_compromising_scores(self, global_update, weight_update):
        cs_dist = get_cs_on_base_net(weight_update, global_update)
        score = np.array(cs_dist)
        norm_score = min_max_scale(score)
        return norm_score
