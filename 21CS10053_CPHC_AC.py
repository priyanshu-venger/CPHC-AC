#Priyanshu kumar
#21cs10053
import sys
import pandas as pd
import numpy as np
from operator import itemgetter
from time import time
from bisect import bisect,bisect_left
start=time()
data=pd.read_csv(sys.stdin)
for column in data: 
    # print(list(data[column]))
    data[column]=data[column].fillna(data[column].mean())
    
data=np.array(data.iloc[:,1:],dtype=np.int64)
optimal_k=3
def cosine_sim(u,v):
    return np.dot(u,v)/(np.sqrt(np.dot(u,u))*np.sqrt(np.dot(v,v)))
def dist(u,v):
    return 1-cosine_sim(u,v)
def save(k,clusters,type):
    if type==0:
        with open("kmeans.txt", "w") as f:
            for i in range(k):
                f.write(f'''{",".join(str(e) for e in (clusters[i]))}\n''')
    else:
        with open("agglomerative.txt", "w") as f:
            for i in range(k):
                f.write(f'''{",".join(str(e) for e in (clusters[i]))}\n''')
def silhoutte_coeff(clusters,cluster_mean,k):
    sc=[]
    for i in range(k):
        w=np.array([dist(cluster_mean[i],cluster_mean[j]) for j in range(k) if i!=j])
        min_loc=np.argmin(w)
        if min_loc>=i:
            min_loc+=1
        for j in range(len(clusters[i])):
            a=np.mean(np.array([dist(data[clusters[i][j]],data[clusters[i][t]]) for t in range(len(clusters[i])) if t!=j]))
            b=np.mean(np.array([dist(data[clusters[i][j]],data[clusters[min_loc][t]]) for t in range(len(clusters[min_loc]))]))
            sc.append((-a+b)/max(a,b))
    return np.mean(np.array(sc))

class k_means:
    def __init__(self,k):
        self.clusters_mean=[0]*k
        for i in range(k):
            self.clusters_mean[i]=data[i]
    def run(self,k,m_iter):
        clusters=[0]*k
        for t in range(m_iter):
            for j in range(k):
                clusters[j]=[]
            for i in range(len(data[:])):
                p=np.argmin(np.array([dist(data[i],self.clusters_mean[j]) for j in range(k)]))
                clusters[p].append(i)
            for j in range(k):
                self.clusters_mean[j]=np.mean(data[clusters[j]],axis=0)
        clusters.sort(key=itemgetter(0))
        return clusters
class CLAC:
    def __init__(self):
        self.buckets=[0]*len(data)
        for i in range(len(data)):
            self.buckets[i]=[]
            self.buckets[i].append(i)
    def run(self,k):
        n=len(data)
        pre_comp_dist=np.zeros(shape=(n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                q=dist(data[i],data[j])
                pre_comp_dist[j][i]=pre_comp_dist[i][j]=q     
        while(n>k):
            min1=np.inf
            for i in range(n-1):
                for j in range(i+1,n):
                    max1=max([pre_comp_dist[p1][p2] for p1 in self.buckets[i] for p2 in self.buckets[j]])
                    if(max1<min1):
                        min_loc1=i
                        min_loc2=j
                        min1=max1
                        
            self.buckets[min_loc1]=self.buckets[min_loc1]+self.buckets[min_loc2]
            self.buckets.pop(min_loc2)
            n-=1
        for i in range(k):
            self.buckets[i]=sorted(self.buckets[i])
        self.buckets.sort(key=itemgetter(0))
        return self.buckets
    def optimized_run(self,k):
        n=len(data)
        pre_comp_dist=np.zeros(shape=(n,n))
        min_tree=[0]*n
        for i in range(n):
            pre_comp_dist[i]=[0]*n
            min_tree[i]=list(tuple())
        for i in range(n-1):
            for j in range(i+1,n):
                if i!=j:
                    pre_comp_dist[j][i]=pre_comp_dist[i][j]=dist(data[i],data[j])
                    min_tree[i].insert(bisect(min_tree[i],(pre_comp_dist[i][j],j)),(pre_comp_dist[i][j],j))
                    min_tree[j].insert(bisect(min_tree[j],(pre_comp_dist[i][j],i)),(pre_comp_dist[i][j],i))
        m=n
        while(m>k):
            min1=np.inf
            for i in range(n):
                if self.buckets[i][0]!=-1:
                    x=min_tree[i][0]
                    if min1>x[0]:
                        min1=x[0]
                        min_loc1=i
                        min_loc2=x[1]
            self.buckets[min_loc1]=self.buckets[min_loc1]+self.buckets[min_loc2]
            for i in range(n):
                if self.buckets[i][0]!=-1 and i!=min_loc1:
                    del min_tree[min_loc1][bisect_left(min_tree[min_loc1],(pre_comp_dist[min_loc1][i],i))]
                    if i!=min_loc2:
                        del min_tree[i][bisect_left(min_tree[i],(pre_comp_dist[i][min_loc2],min_loc2))]
                        del min_tree[i][bisect_left(min_tree[i],(pre_comp_dist[i][min_loc1],min_loc1))]
                        min_tree[i].insert(bisect(min_tree[i],(max(pre_comp_dist[i][min_loc1],pre_comp_dist[i][min_loc2]),min_loc1)),(max(pre_comp_dist[i][min_loc1],pre_comp_dist[i][min_loc2]),min_loc1))
                        pre_comp_dist[i][min_loc1]=max(pre_comp_dist[i][min_loc1],pre_comp_dist[i][min_loc2])
                        pre_comp_dist[i][min_loc2]=-1
                     
            for i in range(n):
                if self.buckets[i][0]!=-1 and i!=min_loc1 and i!=min_loc2:
                    pre_comp_dist[min_loc1][i]=max(pre_comp_dist[min_loc1][i],pre_comp_dist[min_loc2][i])
                    min_tree[min_loc1].insert(bisect(min_tree[min_loc1],(pre_comp_dist[i][min_loc1],i)),(pre_comp_dist[i][min_loc1],i))
            pre_comp_dist[min_loc1][min_loc2]=0
            self.buckets[min_loc2]=[-1]*1
            m-=1
        buckets=[res for res in self.buckets if res[0]!=-1]
        for i in range(k):
            buckets[i]=sorted(buckets[i])
        buckets.sort(key=itemgetter(0))
        return buckets
def jaccard(a,b,k):
    res=[0]*k
    t=0
    for i in a:
        max1=0
        for j in b:
            union=len(set(i).union(set(j)))
            inter=len(set(i).intersection(set(j)))
            max1=max(max1,inter/union)
        res[t]=max1
        t+=1
    return res
def main():
    global data
    # n=int(input("Which agglomerative algorithm you want to check(1:O(n^3),2:(O(n^2log(n):uses bisect library which is not an ML library)):"))  
    n=2 
    if n==1:
        print("Since you've chosen O(n^3) algo,size has been set to 1000")
        data=data[0:1000]
    start=time()
    km=k_means(3)
    opt_cluster=clusters=km.run(3,20)
    max1=silhoutte_coeff(clusters,km.clusters_mean,3)
    print("For k=3,silhoutte_coeff=",max1)
    km=k_means(4)
    clusters=km.run(4,20)
    k=silhoutte_coeff(clusters,km.clusters_mean,4)
    print("For k=4,silhoutte_coeff=",k)
    if k>=max1:
        optimal_k=4
        opt_cluster=clusters
        max1=k
    km=k_means(5)
    clusters=km.run(5,20)
    k=silhoutte_coeff(clusters,km.clusters_mean,5)
    print("For k=5,silhoutte_coeff=",k)
    if k>=max1:
        optimal_k=5
        opt_cluster=clusters
        max1=k
    km=k_means(6)
    clusters=km.run(6,20)
    k=silhoutte_coeff(clusters,km.clusters_mean,6)
    print("For k=6,silhoutte_coeff=",k)
    if k>=max1:
        optimal_k=6
        opt_cluster=clusters
        max1=k
    print("It is max for k=",optimal_k)
    # print(time()-start)
    save(optimal_k,opt_cluster,0)
    start=time()
    clac=CLAC()
    if(n==1):
        agglo=clac.run(optimal_k)
        # clac=CLAC()
        # agglo1=clac.optimized_run(optimal_k)
    else:
        agglo=clac.optimized_run(optimal_k)
    # print(agglo)
    # print(time()-start)
    save(optimal_k,agglo,1)
    print("jaccard score:",jaccard(agglo,opt_cluster,optimal_k))
    # print("jaccard score:",jaccard(agglo,agglo1,optimal_k))
if __name__=="__main__": 
    main()
    # print("runtime:",time()-start)