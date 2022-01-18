# Community-Detection-with-Node2Vec
EE447 course project. Community detection using Node2vec and K-means++ 
## Approach
First we map the nodes into a vector space by Node2vec, then we use K-means to find clusters, whose corresponding nodes constitute communities in the original graph.
Any other clustering algorithm is valid.
## Experiments
We test this algorithm on a community detection task, achieveing 95% accuracy. The algorithm finishes in ten minutes on a graph with 30,000 nodes.
![](https://github.com/rikosellic/Community-detection-with-Node2Vec/blob/main/simple_result.png)
