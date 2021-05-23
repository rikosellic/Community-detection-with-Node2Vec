#!/usr/bin/python
# -*- coding: utf-8 -*-

from node2vec import Node2vec
from kmeans import Kmeans
import csv
from pyvis.network import Network

class Community_detection:
    def __init__(self,edge_location,embedding_location='data/embedding.txt',cluster_location='data/clustering_result.csv',K=5,p=1,q=0.5):
        self.edge_location=edge_location
        self.embedding_location=embedding_location
        self.cluster_location=cluster_location
        self.emb={}
        self.cluster={}
        self.G=[]
        self.node_num=0
        self.K=K
        self.p=p
        self.q=q

    def learn_and_detect(self):
        embedder=Node2vec(p=self.p,q=self.q)
        embedder.learn_and_save(input=self.edge_location,output=self.embedding_location)
        self.emb=embedder.load_embedding(self.embedding_location)
        clustering=Kmeans(K=self.K)
        clustering.load_data_with_dict(self.emb)
        self.cluster=clustering.cluster(self.cluster_location)

    def load_emb_and_detect(self):
        embedder = Node2vec()
        self.emb = embedder.load_embedding(self.embedding_location)
        clustering = Kmeans(K=self.K)
        clustering.load_data_with_dict(self.emb)
        self.cluster=clustering.cluster(self.cluster_location)

    def load_edge(self):
        csv_file = open(self.edge_location,'r')
        csv_reader = csv.reader(csv_file)
        max_node = 0
        for i, row in enumerate(csv_reader):
            source = row[0]
            target = row[1]
            if i == 0:
                continue
            if source != '':
                source = int(source)
                target = int(target)
                max_node = max(max_node, source, target)
                self.G.append((source,target))
        csv_file.close()
        self.nodes_num=max_node+1
        print("{} nodes, {} edges".format(self.nodes_num,len(self.G)))

    def load_clustering(self):
        csv_file = open(self.cluster_location, 'r')
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            id = row[0]
            category =  row[1]
            if i == 0:
                continue
            if id != '':
                id=int(id)
                category=int(category)
                self.cluster[id]=category
        csv_file.close()

    def visualize(self):
        print("Visualizing...")
        colors = ['#fe4365', '#fc9d9a', '#f9cdad','#c8c8a9','#83af9b']
        net=Network('800px','800px')
        for i in range(self.nodes_num):
            net.add_node(i,color=colors[self.cluster[i]])
        for u,v in self.G:
            net.add_edge(u,v)
        net.show_buttons(filter_=['physics'])
        net.show('result.html')
        print('Done')



if __name__=='__main__':
    solver=Community_detection('data/simple.csv',embedding_location='data/simple_embedding.txt',cluster_location='data/simple_clustering_result.csv')
    solver.load_edge()
    #solver.learn_and_detect()
    solver.load_clustering()