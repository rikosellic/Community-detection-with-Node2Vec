#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import math
import random
import time
# This code implements K-means++

class Kmeans:
    def __init__(self,K=5):
        self.dataset=[]
        self.nums=0
        self.K=K
        self.dimension=0

    # Euclidean distance
    def D(self,l1, l2):
        res = 0
        for i in range(self.dimension):
            res += (l1[i] - l2[i]) * (l1[i] - l2[i])
        return math.sqrt(res)

    def load_data_with_dict(self,dic):
        self.nums=len(dic.keys())
        self.dimension=len(dic[0])
        for i in range(len(dic.keys())):
            self.dataset.append(dic[i])
        print('Data Loaded. ')

    def cluster(self,output='data/clustering_result.csv'):
        start=time.time()
        print('Choosing initial points...')
        cluster_centroids=[]
        cluster_ids=[]
        state=[-1 for i in range(self.nums)]

        init_point=random.randint(0,self.nums)
        cluster_centroids.append(self.dataset[init_point])
        cluster_ids.append([init_point])
        state[init_point]=0

        candidates=[9999999 for i in range(self.nums)]
        while len(cluster_ids)<self.K:
            candidates= [min(float(candidates[i]),self.D(self.dataset[i],cluster_centroids[-1])) for i in range(self.nums)]
            next_one=candidates.index(max(candidates))
            cluster_centroids.append(self.dataset[next_one])
            cluster_ids.append([next_one])
            state[next_one]=len(cluster_ids)-1
        print('Initial points chosen! Start K-means...')

        change=True
        counter=0
        while change:
            change = False
            counter+=1
            for id in range(self.nums):
                coordinate=self.dataset[id]
                distances=[self.D(coordinate,centroid_cor) for centroid_cor  in cluster_centroids]
                new_cluster=distances.index(min(distances))
                if state[id]!=new_cluster:
                    change=True
                    if state[id]!=-1:
                        cluster_ids[state[id]].remove(id)
                    state[id]=new_cluster
                    cluster_ids[new_cluster].append(id)
            for c in range(self.K):
                new_cor=[0 for i in range(100)]
                numbers=len(cluster_ids[c])
                for PID in cluster_ids[c]:
                    PID_cor=self.dataset[PID]
                    new_cor=[new_cor[i]+PID_cor[i] for i in range(self.dimension)]
                new_cor=[new_cor[i]/numbers for i in range(self.dimension)]
                cluster_centroids[c]=new_cor
        end=time.time()
        print('K-means finished! Number of loops:{}. Time: {}'.format(counter,end-start))
        print('Writing result...')
        radius=[-9999999 for i in range(self.K)]
        for i in range(self.K):
            for id in cluster_ids[i]:
                radius[i]=max(float(radius[i]),self.D(cluster_centroids[i],self.dataset[id]))
        #sorted_radius=radius.copy()
        #sorted_radius.sort()
        #mapping=[sorted_radius.index(r) for r in radius]

        result={}
        csv_file=open(output,'w',newline='')
        csv_writer=csv.writer(csv_file)
        csv_writer.writerow(['id','category'])
        for i in range(self.nums):
            csv_writer.writerow([i,state[i]])
            result[i]=state[i]
        csv_file.close()
        print('Done')
        return result