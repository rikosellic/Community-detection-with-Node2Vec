#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import random
import time
from gensim.models import Word2Vec

class Node2vec:
    def __init__(self,p=1,q=0.5):
        self.p=p
        self.q=q
        self.adjacent_list={} # Adjacent list, undirected
        self.transition_prob={} #Transsion probability table,  key:(上一个，当前)
        self.walk_length=80
        self.num_walks=10
        self.nodes_num=0
        self.window_size=10
        self.dimension=100

    def load_data(self,input):
        csv_file=open(input)
        csv_reader=csv.reader(csv_file)
        edge_counter=0
        max_node=0
        for i,row in enumerate(csv_reader):
            source=row[0]
            target=row[1]
            if i==0:
                continue
            if source!='':
                edge_counter += 1
                source=int(source)
                target=int(target)
                max_node=max(max_node,source,target)
                if source not in self.adjacent_list:
                    self.adjacent_list[source]=[target]
                elif target not in self.adjacent_list[source]:
                    self.adjacent_list[source].append(target)
                if target not in self.adjacent_list:
                    self.adjacent_list[target]=[source]
                elif source not in self.adjacent_list[target]:
                    self.adjacent_list[target].append(source)
        csv_file.close()
        for i in range(max_node):
            if i not in self.adjacent_list.keys():
                self.adjacent_list[i]=[]
        self.nodes_num=max_node+1
        print('{}nodes,{}edges'.format(max_node+1,edge_counter))
        print('Data loaded.')



    def get_normalized_transition_probability(self,t,v): #t上一个, v当前
        prob = []
        if self.adjacent_list[v]==[]:
            return prob
        for x in self.adjacent_list[v]:  # 下一个
            if x == t:
                prob.append(1 / self.p)
            elif x in self.adjacent_list[t]:
                prob.append(1)
            else:
                prob.append(1 / self.q)
        Z = sum(prob)
        prob = [p / Z for p in prob]
        total=prob[0]
        for i in range(1,len(prob)-1):
            total+=prob[i]
            prob[i]=total
        prob[-1]=1
        return prob

    def preprocess_transition_probs(self):
        print('Calculating transition probability...')
        for v in self.adjacent_list.keys(): #当前
            self.transition_prob[(v,v)]=self.get_normalized_transition_probability(v,v) #第一步
            for t in self.adjacent_list[v]: #上一步
                self.transition_prob[(t,v)]=self.get_normalized_transition_probability(t,v)
        print('Calculated.')


    def walk(self,u,l): #u:起始节点, l:长度
        walk_result=[u]

        if self.adjacent_list[u]==[]:
            for walk_iter in range(l):
                walk_result.append(u)
            return walk_result

        last=walk_result[-1]
        for walk_iter in range(l):
            curr=walk_result[-1]
            choice_number = random.random()
            tp = self.transition_prob[(last, curr)]
            for i in range(len(tp)):
                if choice_number <= tp[i]:
                    walk_result.append(self.adjacent_list[curr][i])
                    last=curr
                    break
        return walk_result

    def simulate_walks(self):
        walks=[]
        nodes=[i for i in range(self.nodes_num)]
        print('Start Walking...')
        for walk_iter in range(self.num_walks):
            print('Walking iteration{}'.format(walk_iter+1))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(node,self.walk_length))
        return walks

    def learn_embedding(self,output):
        walks=self.simulate_walks()
        walks = [list(map(str, walk)) for walk in walks]
        start=time.time()
        print('Start Training....')
        model = Word2Vec(walks, size=self.dimension, window=self.window_size, min_count=0, sg=1, workers=4,
                         iter=10)
        model.wv.save_word2vec_format(output)
        end=time.time()
        print('Training Done. Time:{}'.format(end-start))

    def learn_and_save(self,input,output='data/embedding.txt'):
        self.load_data(input)
        self.preprocess_transition_probs()
        self.learn_embedding(output)

    def load_embedding(self,input='data/embedding.txt'):
        f=open(input,'r')
        self.emb={}
        lines=f.readlines()
        for line in lines[1:]:
            line=line.strip()
            line=line.split()
            id=int(line[0])
            line=line[1:]
            line=list(map(float,line))
            self.emb[id]=line
        f.close()
        return self.emb