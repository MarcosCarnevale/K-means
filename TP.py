# Importando bibliotecas 
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


#Criando Conjunto de dados
X, y = make_blobs(n_samples=500, centers=20, random_state=999)
plt.scatter(X[:,0], X[:,1]) #plotando dados


# optimal_number_of_clusters ưe uma função definida para avaliarmos o melhor numero de clausters para nosso conjunto de dados 
def optimal_number_of_clusters(wcss):
    x1, y1 = 1, wcss[0]
    x2, y2 = 19, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 1

wcss = [] # Lista de inertia (Sum of squared distances of samples to their closest cluster center)

#
for i in range(1, 20):
                    #n_clusters (nunmero de clusters/centroides)
                    #init (Modo de inicialização dos clusters)
                    #max_iter = numero de iterações que o kmeans ira fazer
                        #entenda como iteração o ajuste dos centroides de acordo com a média dos pontos do cluster
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10) #Iniciação do metodo K-means derntro de um for de clusters = 20x
    kmeans.fit(X) # Aplicando k-means aos dados
    wcss.append(kmeans.inertia_) # fazendo append das inertias na lista wcss (inertia = x)
n = optimal_number_of_clusters(wcss) # uso da função optimal_number_of_clusters
print(n) # print do numero ideal de clusters

wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


#plotagem do grafico
plt.plot(range(1, 20), wcss)
plt.plot([1, 19],[wcss[0], wcss[len(wcss)-1]])
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()
