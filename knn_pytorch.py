import os
import sys
import unittest

import torch
from torch.autograd import Variable, Function
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from gather_nn_cuda import gather_nn

class KNearestNeighbor(Function):
    """ Compute k nearest neighbors for each query point.
    """
    def __init__(self, k):
        self.k = k
    '''
    B: batch size
    D: dimension size
    N: number of ref points
    M: number of points to be queried
    K: number of nearest neighbors
    Input
        ref     B,D,N
        query   B,D,M
    Return:
        ind     B,N,K
        feat    B,D,K,N
    '''
    def forward(self, ref, query=None):
        # B,D,N -> B,N,D
        ref_t = torch.transpose(ref, 1, 2)
        ref = ref.float().cuda()
        ref_t = ref_t.float().cuda()
        ref_t2 = torch.sum(ref_t * ref_t, dim=2, keepdim=True)
        if query is not None:
            query_t = torch.transpose(query, 1, 2)
            query = query.float().cuda()
            query_t = query_t.float().cuda()
            query2 = torch.sum(query * query, dim=1, keepdim=True)
        else:
            query = ref
            query_t = ref_t
            query2 = torch.transpose(ref_t2, 1, 2)
        # B,N,D * B,D,M
        m = torch.bmm(ref_t, query)
        # B,N,1 - B,N,M + B,1,M
        dist = ref_t2 - 2 * m + query2
        # B,N,K
        top_ind = torch.topk(dist, self.k, largest=False, dim=2)[1].long().cuda()

        # B,N,K,D
        feat = torch.empty(ref.size(0), ref.size(2), self.k, ref.size(1)).float().cuda().contiguous()
        gather_nn(ref_t.contiguous(), query_t.contiguous(), top_ind.contiguous(), feat)
        # B,N,K,D -> B,D,K,N
        feat = torch.transpose(feat, 1, 3)

        return feat, top_ind

    def backward(self, grad, retain_graph):
        output_grad = torch.mean(grad, dim=2, keepdim=False)
        return output_grad


"""
python version gather_nn, very slow....
Input
    ref     B,D,N
    query   B,D,M
    ind     B,N,K
Output
    feat    B,D,K,N
"""
def py_gather_nn(ref, query, ind):
    query = query.cpu()
    ind = ind.cpu()
    B = ind.size(0)
    N = ind.size(1)
    K = ind.size(2)
    # B,x,D
    ref = torch.transpose(ref, 1, 2)
    query = torch.transpose(query, 1, 2)
    # B,N,K,D
    feat = torch.empty(ind.size(0), ind.size(1), ind.size(2), query.size(2)).float()
    for iB in range(B):
        for iN in range(N):
            for i in range(K):
                feat[iB, iN, i] = query[iB, ind[iB, iN, i]] - ref[iB, iN]
    # B,N,K,D -> B,D,K,N
    feat = torch.transpose(feat, 1, 3)
    return feat

class TestKNearestNeighbor(unittest.TestCase):
    def test_forward1(self):
        B, D, N, M = 5, 4, 6500, 6500
        k = 16
        for _ in range(10):
            ref = Variable(torch.randn(B, D, N, dtype=torch.float))
            query = Variable(torch.randn(B, D, M, dtype=torch.float))
            print('*', end='', flush=True)
            feat, ind = KNearestNeighbor(k)(ref, query)
            pyfeat = py_gather_nn(ref, query, ind)
            assert torch.equal(pyfeat, feat.cpu()), "python & cuda version not match"

    def test_forward2(self):
        # 1, 2, 3
        ref = Variable(torch.tensor([[[0,10,100], [0,10,100]]], dtype=torch.float))
        # 1, 2, 6
        query = Variable(torch.tensor([[[1,2,11,12,101,102], [1,2,11,12,101,102]]], dtype=torch.float))
        feat, ind = KNearestNeighbor(4)(ref, query)
        print('\n{}'.format(ind.cpu().numpy()))
        print('\n{}'.format(feat.cpu().numpy()))
        pyfeat = py_gather_nn(ref, query, ind)
        print('\n{}'.format(pyfeat.cpu().numpy()))
        assert torch.equal(pyfeat, feat.cpu()), "python & cuda version not match"

    def test_forward3(self):
        B, D, N = 5, 10, 6500
        k = 16
        for _ in range(10):
            ref = Variable(torch.randn(B, D, N, dtype=torch.float))
            print('*', end='', flush=True)
            feat, ind = KNearestNeighbor(k)(ref)
            pyfeat = py_gather_nn(ref, ref, ind)
            assert torch.equal(pyfeat, feat.cpu()), "python & cuda version not match"

if __name__ == '__main__':
    unittest.main(verbosity=2)
# [[[0 1 2 3]
#   [2 3 1 0]
#   [4 5 3 2]]]

# [[[[  1.  11. 101.]
#    [  2.  12. 102.]
#    [ 11.   2.  12.]
#    [ 12.   1.  11.]]

#   [[  1.  11. 101.]
#    [  2.  12. 102.]
#    [ 11.   2.  12.]
#    [ 12.   1.  11.]]]]
# [[[0 1 2 3]
#   [2 3 1 0]
#   [4 5 3 2]]]

# [[[[  1.         11.        101.       ]
#    [ -8.          2.         92.       ]
#    [-89.        -98.        -88.       ]
#    [ 11.96706     0.9670604  10.96706  ]]

#   [[  1.         11.        101.       ]
#    [ -8.          2.         92.       ]
#    [-89.        -98.        -88.       ]
#    [ 11.941363    0.9413633  10.941363 ]]]]
