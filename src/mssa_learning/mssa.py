#!/usr/bin/env python
import numpy as np
from mssa_learning.tools.preprocessing import postural_distance


class MSSA(object):
    """docstring for MSSA"""
    def __init__(self):
        super(MSSA, self).__init__()
        self.M = None
        self.N = None
        self.D = None

    def _compute_covariance_matrix(self, data):
        assert(self.N is not None)
        assert(self.D is not None)
        # compress the signal on the time window
        y = np.zeros((self.N - self.M+1, self.M))
        for m in range(self.M):
            y[:, m] = data[0, m:self.N-self.M+1+m]
        Y = y
        for d in range(1, self.D):
            y = np.zeros((self.N - self.M+1, self.M))
            for m in range(self.M):
                y[:, m] = data[d, m:self.N-self.M+1+m]
            Y = np.hstack((Y, y))
        # calculate the covariance
        Cemb = np.dot(np.transpose(Y),Y) / (self.N-self.M+1)
        return Y, Cemb

    @staticmethod
    def _compute_eigenbasis(covar):
        eig_val,eig_vec = np.linalg.eigh(covar)
        indexes = np.argsort(eig_val)[::-1]
        eig_val = np.array([eig_val[i] for i in indexes])
        eig_vec = np.transpose([eig_vec[:, i] for i in indexes])
        return eig_val, eig_vec

    def compute_principal_components(self, data, M=None):
        self.N = len(data[0])
        self.D = len(data)
        if M is None:
            self.M = int(self.N / 2)
        else:
            self.M = M

        # compute the covariance
        Y, covar = self._compute_covariance_matrix(data)
        # compute the eigenbasis
        eig_val, eig_vec = self._compute_eigenbasis(covar)
        # finally the principal components
        PC = np.dot(Y, eig_vec)
        return PC, eig_vec, eig_val

    def _compute_recontstructed_components(self, PC, eig_vec, nb_components):
        RC = []
        for d in range(self.D):
            rc = np.zeros((self.N, self.D*self.M))
            for m in range(nb_components):
                buf = np.dot(np.transpose([PC[:, m]]), [np.transpose(eig_vec[d*self.M:(d+1)*self.M, m])])
                buf = np.flipud(buf)
                for n in range(self.N):
                    rc[n, m] = np.mean(np.diag(buf, -(self.N-self.M) + n))

            RC.append(rc)
        return RC

    def reconstruct_signal(self, PC, eig_vec, nb_components):
        rs = []
        RC = self._compute_recontstructed_components(PC, eig_vec, nb_components)
        for rc in RC:
            rs.append(np.sum(rc, 1))
        return np.array(rs)

    def signal_distance(self, data, rs_data):
        dist = 0
        for i in range(len(data[0])):
            dist += postural_distance(data[:, i], rs_data[:, i])
        return dist
