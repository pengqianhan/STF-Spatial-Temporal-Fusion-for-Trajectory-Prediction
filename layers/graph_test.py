import numpy as np


class Graph():
    """ The Graph Representation
	How to use:
		1. graph = Graph(max_hop=1)
		2. A = graph.get_adjacency()
		3. A = code to modify A
		4. normalized_A = graph.normalize_adjacency(A)
	"""

    def __init__(self,
                 num_node=120,
                 max_hop=2
                 ):
        self.max_hop = max_hop
        self.num_node = num_node

    def get_adjacency(self, A):
        # compute hop steps
        self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf  ## np.inf= 正无穷
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]  ## 计算矩阵的次方
        # print('transfer_mat',transfer_mat)
        arrive_mat = (np.stack(transfer_mat) > 0)  ##
        print('arrive_mat \n', arrive_mat)
        # print('arrive_mat shape',arrive_mat)##(2,3,3)

        for d in range(self.max_hop, -1, -1):
            print('arrive_mat[{}] \n {}'.format(d, arrive_mat[d]))
            print('self.hop_dis shape:', (self.hop_dis).shape)
            self.hop_dis[arrive_mat[d]] = d
            print('for self.hop_dis:\n', self.hop_dis)  ##

        print('self.hop_dis\n', self.hop_dis)  ##
        # compute adjacency
        valid_hop = range(0, self.max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            print('self.hop_dis{}\n {} '.format(hop, self.hop_dis))
            adjacency[self.hop_dis == hop] = 1
            print('adjacency\n', adjacency)
        return adjacency

    def normalize_adjacency(self, A):

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                print('Dl[i]', Dl[i])
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)  ##A*D^{-1}
        print('AD\n', AD)
        valid_hop = range(0, self.max_hop + 1)  ###max_hop=1
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            print(i, hop)
            print('self.hop_dis \n', self.hop_dis)
            A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
			# print('A:',A)

        return A  ##

A=np.ones((3,3))
A[0, 1]=0
A[1, 1]=0
A[2,2]=0
print(A)
graphfuction=Graph(num_node = A.shape[0],max_hop=1)

out=graphfuction.get_adjacency(A)
print('out',out)

out_n=graphfuction.normalize_adjacency(out)
print('out_n',out_n)

