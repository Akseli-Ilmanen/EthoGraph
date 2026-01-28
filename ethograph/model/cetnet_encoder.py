import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import Tensor
import copy
import numpy as np
import math
from typing import Tuple
import xarray as xr
import os
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import entropy
from ethograph.utils.io import TrialTree
from ethograph.model.eval_metrics import func_eval, func_eval_labelwise
from ethograph.features.changepoints import correct_changepoints_one_trial
from ethograph.utils.labels import create_classification_probabilities_pdf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss



class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6)  # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()

    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i:i + self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder (not used)

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        value = self.value_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                                  torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)], dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,
                                                                                                     1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                                  torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)], dim=-1)

        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k,
                       torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v,
                       torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask,
                                  torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)

        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)],
                      dim=0)  # special case when self.bl = 1
        v = torch.cat([v[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)], dim=0)
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)],
            dim=0)  # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask

        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        # RF = 1 + 2 * dilation, where dilation = 2**i, i is the layer index, e.g. layer 4 -> 1 + 2 * 2**4 = 33
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation), # kernel size = 3 (hard coded)
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
                                  stage=stage)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]



class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in  # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature



class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,
                               att_type='sliding_att', alpha=1)


    # We removed decoder, but still pass encoder unsqueezed outputs for compatibility with previous code.
    def forward(self, x, mask):
        encoder_out, feature = self.encoder(x, mask)
        
        return encoder_out.unsqueeze(0), F.normalize(feature, dim=1).unsqueeze(0)



class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, f1_thresholds, boundary_weight_schedule, boundary_radius):
        self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.num_f_maps =num_f_maps
        self.floss = CircleLoss(m=0.25, gamma=128)
        self.f1_thresholds = f1_thresholds
        self.boundary_weight = 0.0 # Will be updated by schedule
        self.boundary_weight_schedule = boundary_weight_schedule
        self.boundary_radius = boundary_radius
        


    def compute_boundary_mask(self, target, radius=2):
        transitions = (target[:, 1:] != target[:, :-1]).float()
        transitions = F.pad(transitions, (1, 1), value=0)
        kernel = torch.exp(-torch.abs(torch.arange(-radius, radius+1, device=target.device)) / radius)
        kernel = kernel.view(1, 1, -1)
        transitions = transitions.unsqueeze(1)
        boundary_mask = F.conv1d(transitions, kernel, padding=radius).squeeze(1)
        boundary_mask = boundary_mask[:, :target.shape[1]]
        
        boundary_mask = torch.clamp(boundary_mask * self.boundary_weight, max=2.0) # Clamp for Conv padding effect
        return boundary_mask


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None, all_params=None):
        self.model.train()
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR:{}'.format(learning_rate))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_mse_loss = 0
            correct = 0
            total = 0
            
            
            self.boundary_weight = max(
                (w for t, w in self.boundary_weight_schedule.items() if epoch >= int(t)),
                default=0.0
            )


            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input = batch_input.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                optimizer.zero_grad()
                ps, features = self.model(batch_input, mask)
                loss = 0

                boundary_mask = self.compute_boundary_mask(batch_target, self.boundary_radius)


                # ps is only 1 element list here -> encoder only
                for idx, p in enumerate(ps):
                    ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    ce_loss = ce_loss.view(batch_target.shape)
                    


                    # 1* (no weighting), after epoch 40 1-3* (boundary weighting)
                    weighted_ce = ce_loss * (1.0 + boundary_mask * mask[:, 0, :])
                    loss += torch.mean(weighted_ce * mask[:, 0, :])                        
                    epoch_ce_loss += torch.mean(ce_loss * mask[:, 0, :]).item()


                    mse_loss = torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), 
                                F.log_softmax(p.detach()[:, :, :-1], dim=1)), 
                        min=0, max=16) * mask[:, :, 1:])
                    loss += 0.15 * mse_loss
                    
                    epoch_mse_loss += mse_loss.item()       
     

     
     
     
                if all_params.get("no_circle_loss", True):
                    for f in features:
                        loss += 0.001 * self.floss(
                            *convert_label_to_similarity(f.transpose(2, 1).contiguous().view(-1, self.num_f_maps),
                                                        batch_target.view(-1)))

                    epoch_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
                optimizer.step()

                _, predicted = torch.max(ps.data[0], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            scheduler.step(epoch_loss)
            batch_gen.reset()
            # from thop import profile
            #
            # flops, params = profile(self.model, inputs=(batch_input, mask))
            # print("flops:", flops)
            # print("params:", params)
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                            float(correct) / total))
            print("    ce_loss = %f, mse_loss = %f" % (
                epoch_ce_loss / len(batch_gen.list_of_examples),
                epoch_mse_loss / len(batch_gen.list_of_examples)
            ))    
            


            if (epoch + 1) % all_params.get("log_freq") == 0 and batch_gen_tst is not None or epoch == 0:
                self.test(batch_gen_tst, epoch+1, all_params)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")


       
    def test(self, batch_gen_tst, epoch, all_params):
        self.model.eval()
        self.model.to(device)
        if_warp = False  # When testing, always false
            
        ground_truth_dict = dict()
        pred_dict = dict()
        corr_pred_dict = dict()
        video_list = []
        previous_hash = None
        trial_mapping = json.load(open(os.path.join(all_params["dataset_dir"], 'trial_mapping.json')))
        
        
        
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                p, _ = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[0], 1)
                predicted = predicted.squeeze().cpu().numpy()
                

                vid = vids[0].split('.')[0]
                ground_truth_dict[vid] = batch_target.squeeze().cpu().numpy()
                pred_dict[vid] = predicted
                video_list.append(vid)
                
                hash_key, trial = vid.split('_')
                
                # Prevent from re-loading the same TrialTree multiple times
                if hash_key != previous_hash:
                    previous_hash = hash_key
                    nc_path = trial_mapping[hash_key]["nc_path"]
                    dt = TrialTree.load(nc_path)
                    

    
                ds = dt.trial(trial)
                corr_pred = correct_changepoints_one_trial(predicted, ds, all_params)               
                corr_pred_dict[vid] = corr_pred
                
                
                


        # Evaluate both uncorrected and corrected predictions
        nested_results = {}
            
        for pred_type, pred_dict in [("uncorrected", pred_dict), ("corrected", corr_pred_dict)]:
            acc, edit, f1s, tp, fp, fn, frame_f1 = func_eval(ground_truth_dict, pred_dict, video_list, self.f1_thresholds)

            
            result_dict = {
                'Acc': acc,
                'Edit': edit,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                "Frame_F1": frame_f1,
            }

            # Add F1 scores dynamically based on f1_thresholds
            for i, threshold in enumerate(self.f1_thresholds):
                result_dict[f'F1@{int(threshold*100)}'] = f1s[i] # =.5, =.75, =.95 -> 50, 75, 95

            classResults, _, all_IoUs, start_deltas, end_deltas = func_eval_labelwise(ground_truth_dict, pred_dict, video_list, f1_thresholds=self.f1_thresholds)

            # Add class-wise F1 scores - classResults is a dict with class keys containing f1s arrays
            result_dict['classwise_results'] = classResults
            result_dict['all_IoUs'] = all_IoUs
            result_dict['start_deltas'] = start_deltas
            result_dict['end_deltas'] = end_deltas
            
            nested_results[pred_type] = result_dict



            result_dir = all_params.get("result_dir")
            np.save(os.path.join(result_dir, f'test_results_epoch{epoch}.npy'), nested_results)

        # Print results for both corrected and uncorrected predictions
        for pred_type, test_result_dict in nested_results.items():
            print(f'Epoch {epoch} ----- -Test-{pred_type}:')
            for k, v in test_result_dict.items():
                if k not in ['all_IoUs', 'start_deltas', 'end_deltas', 'classwise_results']:
                    print(f'  {k}: {v}')


        self.model.train()
        batch_gen_tst.reset()
        
        
        

        

    def inference(self, model_path, features_path, batch_gen_tst, epoch, trial_mapping, sample_rate, all_params):
        self.model.eval()
        

        with torch.no_grad():
            self.model.to(device)
            print("Loading model from {}".format(model_path))
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

            batch_gen_tst.reset(shuffle=False)           

            previous_hash = None    
            sess_dict = {}
            for key in trial_mapping.keys():
                nc_path = trial_mapping[key]["nc_path"]
                dt = TrialTree.load(nc_path)
                pred_dt = dt.get_label_dt(empty=True)
                corr_pred_dt = dt.get_label_dt(empty=True)

                sess_dict[key] = {"pred_dt": copy.deepcopy(pred_dt),
                                  "corr_pred_dt": copy.deepcopy(corr_pred_dt),
                                  "nc_path": nc_path,
                                  "inference": False}
                
                
            print("Running inference...")
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)

                vid = vids[0]
                vid = vid.split('.')[0]
                features = np.load(features_path + vid + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)

                # input_x =input_x.transpose(1, 0)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)

                predictions, features = self.model(input_x, torch.ones(input_x.size(), device=device))
    
                _, predicted = torch.max(predictions[0].data, 1)
                predicted = predicted.squeeze().cpu().numpy()


                probs = torch.softmax(predictions[0], dim=1).cpu().numpy()
                max_entropy = np.log(probs.shape[1])  # Maximum possible entropy
                confidence = 1 - entropy(probs, axis=1) / max_entropy  # Normalized to [0, 1]
                if confidence.ndim > 1:
                    confidence = confidence.squeeze()
            
            
            
                hash_key, trial_num = vid.split('_')
                individual = all_params["target_individual"]
                
                
                pred_dt = sess_dict[hash_key]["pred_dt"]
                pred_dt.trial(trial_num).labels.loc[{"individuals": individual}] = predicted

                pred_dt.new_var_like(
                        trial=trial_num,
                        new_var='labels_confidence',
                        new_val=confidence,
                        template_var='labels',
                        kwargs={'individuals': individual}
                    )
                sess_dict[hash_key]["pred_dt"] = pred_dt
                
                
                corr_pred_dt = sess_dict[hash_key]["corr_pred_dt"]
   
             
                if hash_key != previous_hash:
                    previous_hash = hash_key
                    nc_path = trial_mapping[hash_key]["nc_path"]
                    dt = TrialTree.load(nc_path)
                
                corr_pred = correct_changepoints_one_trial(predicted, dt.trial(trial_num), all_params)      
                
            
                corr_pred_dt.trial(trial_num).labels.loc[{"individuals": individual}] = corr_pred
                
                
                

                corr_pred_dt.new_var_like(
                        trial=trial_num,
                        new_var='labels_confidence',
                        new_val=confidence, # same as uncorrected
                        template_var='labels',
                        kwargs={'individuals': individual}
                    )
                
                sess_dict[hash_key]["corr_pred_dt"] = corr_pred_dt                


                sess_dict[hash_key]["inference"] = True
                
                
                

            
            for key in sess_dict.keys():
                if sess_dict[key].get("inference"):
                    
                    
                    # Paths
                    nc_path_obj = Path(sess_dict[key]["nc_path"])
                    print(f"Saving predictions for session {key}..., nc_path: {nc_path_obj}")
                    labels_dir = nc_path_obj.parent / "labels"
                    labels_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    pred_dt = sess_dict[key]["pred_dt"]
                    versioned_path = labels_dir / f"{nc_path_obj.stem}_predictions_uncorr_{timestamp}{nc_path_obj.suffix}"
                    pred_dt.attrs["changepoint_corrected"] = np.int8(1)          
                    pred_dt.to_netcdf(versioned_path)
                    
                    
                    # Save confidence PDF
                    corr_pred_dt = sess_dict[key]["corr_pred_dt"]
                    
                    
                    pdf_filename = f"{nc_path_obj.stem}_classification_probabilities_{timestamp}.pdf"
                    pdf_path = labels_dir / pdf_filename
                    try:
                        corr_pred_dt = create_classification_probabilities_pdf(corr_pred_dt, pdf_path) # Get confidence attrs
                    except Exception as e:
                        print(f"Warning: Failed to create classification probabilities PDF: {e}")
                    
                    
                    corr_pred_dt.attrs["changepoint_corrected"] = np.int8(1)
                    versioned_path = labels_dir / f"{nc_path_obj.stem}_predictions_corr_{timestamp}{nc_path_obj.suffix}"
                    corr_pred_dt.to_netcdf(versioned_path)


    def inference_stepwise(self, model_path, features_path, batch_gen_tst, epoch, trial_mapping, sample_rate, all_params):
        self.model.eval()
        
        
        with torch.no_grad():
            self.model.to(device)
            print("Loading model from {}".format(model_path))
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

            batch_gen_tst.reset(shuffle=False)           

            previous_hash = None    
            sess_dict = {}
            
            for key in trial_mapping.keys():
                nc_path = trial_mapping[key]["nc_path"]
                dt = TrialTree.load(nc_path)
                pred_dt = dt.get_label_dt(empty=True)
                corr_pred_dt = dt.get_label_dt(empty=True)
                
                sess_dict[key] = {
                    "pred_dt_encoder": copy.deepcopy(pred_dt),
                    "pred_dt_decoder1": copy.deepcopy(pred_dt),
                    "pred_dt_decoder2": copy.deepcopy(pred_dt),
                    "pred_dt_decoder3": copy.deepcopy(pred_dt),
                    "corr_pred_dt": corr_pred_dt,
                    "nc_path": nc_path,
                    "inference": False
                }
                
            print("Running inference...")
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)

                vid = vids[0]
                vid = vid.split('.')[0]
                features = np.load(features_path + vid + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)

                # predictions[0] = encoder output
                # predictions[1:4] = decoder outputs (3 decoders)
                predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device))
                

                
                # Extract all predictions: encoder + 3 decoders
                all_predictions = []
                for i in range(len(predictions)):
                    _, predicted = torch.max(predictions[i].data, 1)
                    predicted = predicted.squeeze().cpu().numpy()
                    
                    
                    all_predictions.append(predicted)
                    
                    
            
                hash_key, trial_num = vid.split('_')
                individual = all_params["target_individual"]
                
                # Store encoder prediction (index 0)
                pred_dt_encoder = sess_dict[hash_key]["pred_dt_encoder"]
                pred_dt_encoder.trial(trial_num).labels.loc[{"individuals": individual}] = all_predictions[0]
                sess_dict[hash_key]["pred_dt_encoder"] = pred_dt_encoder
                
                # Store decoder predictions (indices 1, 2, 3)
                decoder_keys = ["pred_dt_decoder1", "pred_dt_decoder2", "pred_dt_decoder3"]
                for i, decoder_key in enumerate(decoder_keys):
                    if i + 1 < len(all_predictions):
                        pred_dt_decoder = sess_dict[hash_key][decoder_key]
                        pred_dt_decoder.trial(trial_num).labels.loc[{"individuals": individual}] = all_predictions[i + 1]
                        sess_dict[hash_key][decoder_key] = pred_dt_decoder
                
                # Corrected prediction from last decoder
                corr_pred_dt = sess_dict[hash_key]["corr_pred_dt"]
                
                if hash_key != previous_hash:
                    previous_hash = hash_key
                    nc_path = trial_mapping[hash_key]["nc_path"]
                    dt = TrialTree.load(nc_path)
                
                corr_pred = correct_changepoints_one_trial(all_predictions[-1], dt.trial(trial_num), all_params)      
                corr_pred_dt.trial(trial_num).labels.loc[{"individuals": individual}] = corr_pred
                sess_dict[hash_key]["corr_pred_dt"] = corr_pred_dt                

                sess_dict[hash_key]["inference"] = True

            # Save all predictions
            for key in sess_dict.keys():
                if sess_dict[key].get("inference"):
                    nc_path_obj = Path(sess_dict[key]["nc_path"])
                    print(f"Saving predictions for session {key}..., nc_path: {nc_path_obj}")
                    labels_dir = nc_path_obj.parent / "labels"
                    labels_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save encoder predictions
                    pred_dt_encoder = sess_dict[key]["pred_dt_encoder"]
                    versioned_path = labels_dir / f"{nc_path_obj.stem}_predictions_encoder_{timestamp}{nc_path_obj.suffix}"
                    pred_dt_encoder.attrs["changepoint_corrected"] = np.int8(0)
                    pred_dt_encoder.to_netcdf(versioned_path)
                    
                    # Save decoder predictions
                    for i in range(1, 4):
                        decoder_key = f"pred_dt_decoder{i}"
                        pred_dt_decoder = sess_dict[key][decoder_key]
                        versioned_path = labels_dir / f"{nc_path_obj.stem}_predictions_decoder{i}_{timestamp}{nc_path_obj.suffix}"
                        pred_dt_decoder.attrs["changepoint_corrected"] = np.int8(0)
                        pred_dt_decoder.to_netcdf(versioned_path)
                    
                    # Save corrected predictions
                    corr_pred_dt = sess_dict[key]["corr_pred_dt"]
                    corr_pred_dt.attrs["changepoint_corrected"] = np.int8(1)
                    versioned_path = labels_dir / f"{nc_path_obj.stem}_predictions_corr_{timestamp}{nc_path_obj.suffix}"
                    corr_pred_dt.to_netcdf(versioned_path)

                    # Save confidence PDF
                    pdf_filename = f"{nc_path_obj.stem}_classification_probabilities_{timestamp}.pdf"
                    pdf_path = labels_dir / pdf_filename
                    try:
                        create_classification_probabilities_pdf(corr_pred_dt, pdf_path)
                    except Exception as e:
                        print(f"Warning: Failed to create classification probabilities PDF: {e}")                    


