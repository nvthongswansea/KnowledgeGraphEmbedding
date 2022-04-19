#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
from ga import GA


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        # q = GA.tensor_to_mv(torch.tensor([[4, 3, 2, 1],[1,2,3,4]]))
        # print(q)
        # exit(0)
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
            
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        if model_name == 'cliffordRotatE':
            self.entity_dim = 4
            self.relation_dim = 1
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'cliffordRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single', debug_msg='', isEvaluateStep=False):
        # print('*******************DEBUG MESSAGE', debug_msg)
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # print('----------------single----------------')
            batch_size, negative_sample_size = sample.size(0), 1
            if self.model_name == 'cliffordRotatE':
                head = [self.entity_embedding[i] for i in sample[:,0]]
                relation = [self.relation_embedding[i] for i in sample[:,1]]
                tail = [self.entity_embedding[i] for i in sample[:,2]]
            else:
                head = torch.index_select(
                    self.entity_embedding, 
                    dim=0, 
                    index=sample[:,0]
                ).unsqueeze(1)
                
                relation = torch.index_select(
                    self.relation_embedding, 
                    dim=0, 
                    index=sample[:,1]
                ).unsqueeze(1)
                
                tail = torch.index_select(
                    self.entity_embedding, 
                    dim=0, 
                    index=sample[:,2]
                ).unsqueeze(1)
            
        elif mode == 'head-batch':
            # print('----------------head-batch----------------')
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            # print('head-batch')
            # print('pos', tail_part)
            # print('neg', head_part)
            # exit(0)
            if self.model_name == 'cliffordRotatE':
                head = []
                relation = [self.relation_embedding[i] for i in tail_part[:,1]]
                tail = [self.entity_embedding[i] for i in tail_part[:,2]]
                for i in range(len(head_part)):
                    head.append([self.entity_embedding[j] for j in head_part[i,:]])
            else:
                head = torch.index_select(
                    self.entity_embedding, 
                    dim=0, 
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                
                relation = torch.index_select(
                    self.relation_embedding, 
                    dim=0, 
                    index=tail_part[:, 1]
                ).unsqueeze(1)
                
                tail = torch.index_select(
                    self.entity_embedding, 
                    dim=0, 
                    index=tail_part[:, 2]
                ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            # print('----------------tail-batch----------------')
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # print('tail-batch')
            # print('pos', head_part)
            # print('neg', tail_part)
            if self.model_name == 'cliffordRotatE':
                head = [self.entity_embedding[i] for i in head_part[:,0]]
                relation = [self.relation_embedding[i] for i in head_part[:,1]]
                tail = []
                # for i in head_part[:,0]:
                #     head.append(self.entity_embedding[i])
                # for i in head_part[:,1]:
                #     relation.append(self.relation_embedding[i])
                for i in range(len(tail_part)):
                    tail.append([self.entity_embedding[j] for j in tail_part[i,:]])
                # exit(0)
            else:
                
                head = torch.index_select(
                    self.entity_embedding, 
                    dim=0, 
                    index=head_part[:, 0]
                ).unsqueeze(1)
                
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)
                
                tail = torch.index_select(
                    self.entity_embedding, 
                    dim=0, 
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'cliffordRotatE': self.cliffordRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode, isEvaluateStep)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        phase_relation = relation/(self.embedding_range.item()/pi)

        #Make phases of relations uniformly distributed in [-pi, pi]


        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def cliffordRotatE(self, head, relation, tail, mode, isEvaluateStep=False):
                
        pi = 3.14159265358979323846
        score = []
        phase_relation = [rela_i/(self.embedding_range.item()/pi) for rela_i in relation]
        w = [GA.tensor_to_mv(torch.tensor([np.cos(phase_i.item()), 0, 0, -np.sin(phase_i.item())])) for phase_i in phase_relation]
        w_hat = [GA.tensor_to_mv(torch.tensor([np.cos(phase_i.item()), 0, 0, np.sin(phase_i.item())])) for phase_i in phase_relation]
        if isEvaluateStep:
            # print("---------------------------------------------------------------------------------------------------------------------")
            # print("HEAD", head)
            # print("---------------------------------------------------------------------------------------------------------------------")
            # print("RELATION", relation)
            # print("---------------------------------------------------------------------------------------------------------------------")
            # print("TAIL", tail)
            # print("---------------------------------------------------------------------------------------------------------------------")
            if mode == 'head-batch':
                tail_cl = [GA.tensor_to_mv(tail_i) for tail_i in tail]
                for (head_i, w_i, w_hat_i , tail_cl_i) in zip(head, w, w_hat, tail_cl):
                    head_i_cl = [GA.tensor_to_mv(head_j) for head_j in head_i]
                    r_head = [w_i*head_i_cl_j for head_i_cl_j in head_i_cl]
                    r_head = [r_head_i*w_hat_i for r_head_i in r_head]
                    score_cl = [r_head_i-tail_cl_i for r_head_i in r_head]
                    score_i_tensor = []
                    for score_cl_i in score_cl:
                        score_cli_i_tensor = GA.mv_to_tensor(score_cl_i)
                        score_i_tensor.append((score_cli_i_tensor[1] + score_cli_i_tensor[2] + score_cli_i_tensor[3]) / 3)
                    score_i_tensor = torch.vstack(score_i_tensor)
                    score.append(torch.squeeze(score_i_tensor))
                    # print("---------------------------------------------------------------------------------------------------------------------")
                    # print("score_i_tensor", torch.squeeze(score_i_tensor), score_i_tensor.shape)
                    # print("---------------------------------------------------------------------------------------------------------------------")
                    # print("w_i", w_i)
                    # print("---------------------------------------------------------------------------------------------------------------------")
                    # print("w_hat_i", w_hat_i)
                    # print("---------------------------------------------------------------------------------------------------------------------")
                    # print("tail_cl_i", tail_cl_i)
                    # print("---------------------------------------------------------------------------------------------------------------------")
                    # print('score HEAD *****', score, len(score))
            if mode == 'tail-batch':
                head_cl = [GA.tensor_to_mv(head_i) for head_i in head]
                r_head = [w_i*head_cl_i for (w_i,head_cl_i) in zip(w, head_cl)]
                r_head = [head_cl_i*w_hat_i for (head_cl_i,w_hat_i) in zip(r_head, w_hat)]
                print("---------------------------------------------------------------------------------------------------------------------")
                print("r_head", r_head)
                print("---------------------------------------------------------------------------------------------------------------------")
                for (r_head_i, w_i, w_hat_i , tail_i) in zip(r_head, w, w_hat, tail):
                    tail_i_cl = [GA.tensor_to_mv(tail_j) for tail_j in tail_i]
                    print("---------------------------------------------------------------------------------------------------------------------")
                    print("tail_i_cl", tail_i_cl, len(tail_i_cl))
                    print("---------------------------------------------------------------------------------------------------------------------")
                    
            # print("score", score)
            # if len(score) == 0:
            #     print("mode", mode)
            score = torch.vstack(score)
            # print('score ^^^^^^^^^^^', mode, score, score.shape)
            # exit(0)
            score = self.gamma.item() - score
            return score

        # print('----r_head ccccc', [GA.mv_to_tensor(tail_i) for tail_i in w])
        if mode == 'head-batch':
            tail_cl = [GA.tensor_to_mv(tail_i) for tail_i in tail]
            # print('len head', len(head))
            for (head_i, tail_cl_i) in zip(head, tail_cl):
                head_i_cl = [GA.tensor_to_mv(head_j) for head_j in head_i]
                r_head = [w_i*head_i_cl_j for (w_i,head_i_cl_j) in zip(w, head_i_cl)]
                r_head = [r_head_i*w_hat_i for (r_head_i,w_hat_i) in zip(r_head, w_hat)]
                score_cl = [r_head_i-tail_cl_i for (r_head_i,tail_cl_i) in zip(r_head, tail_cl)]
                score_stack = torch.tensor([0.])
                for score_cl_i in score_cl:
                    score_cli_i_tensor = GA.mv_to_tensor(score_cl_i)
                    score_stack += (score_cli_i_tensor[1] + score_cli_i_tensor[2] + score_cli_i_tensor[3]) / 3    
                score_stack = score_stack / len(score_cl)
                score.append(score_stack)
            # print('score HEAD *****', score)
        elif mode == 'tail-batch':
            head_cl = [GA.tensor_to_mv(head_i) for head_i in head]
            r_head = [w_i*head_cl_i for (w_i,head_cl_i) in zip(w, head_cl)]
            r_head = [head_cl_i*w_hat_i for (head_cl_i,w_hat_i) in zip(r_head, w_hat)]
            for (head_cl_i,tail_i) in zip(r_head, tail):
                score_stack = torch.tensor([0.])
                for j in range(len(tail_i)):
                    score_cl = head_cl_i - GA.tensor_to_mv(tail_i[j])
                    score_tensor = GA.mv_to_tensor(score_cl)
                    score_stack += (score_tensor[1] + score_tensor[2] + score_tensor[3]) / 3
                score_stack = score_stack / len(tail_i)
                # print('score_stack',score_stack)
                score.append(score_stack)
            # print('score TAIL ------', score)
        else:
            head_cl = [GA.tensor_to_mv(head_i) for head_i in head]
            r_head = [w_i*head_cl_i for (w_i,head_cl_i) in zip(w, head_cl)]
            r_head = [head_cl_i*w_hat_i for (head_cl_i,w_hat_i) in zip(r_head, w_hat)]
            tail_cl = [GA.tensor_to_mv(tail_i) for tail_i in tail]
            score_cl = [head_cl_i-tail_cl_i for (head_cl_i,tail_cl_i) in zip(r_head, tail_cl)]
            score = [(score_cl_i[1]+score_cl_i[2]+score_cl_i[3])/3 for score_cl_i in score_cl]
            # score_s = torch.stack(score_tensor)
            # score = GA.mv_to_tensor(score_cl)
            # print('cc', head)
            # print('cc', w)
            # print('cc', w_hat)
            # print('score_stack', score_stack)
        score = torch.vstack(score)
        # score = score.sum(dim = 1)
        # print('score ^^^^^^^^^^^', mode, score)

        score = self.gamma.item() - score
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode,debug_msg='negative_score calculation')

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample,debug_msg='positive_score calculation')

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        print("is COUNTRIES used?", args.countries)
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode, isEvaluateStep=True)
                        # print("---------------------------------------------------------------------------------------------------------------------")
                        # print("FILTER_BIAS", filter_bias)
                        # print("---------------------------------------------------------------------------------------------------------------------")
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
