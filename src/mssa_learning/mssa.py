#!/usr/bin/env python
import numpy as np
from scipy.stats import norm


class MSSA(object):
    """docstring for MSSA"""
    def __init__(self, M):
        super(MSSA, self).__init__()
        self.M = M
        self.N = None
        self.D = None

    def _compute_covariance_matrix(self, data, N, D):
        # compress the signal on the time window
        y = np.zeros((N - self.M+1, self.M))
        for m in range(self.M):
            y[:, m] = data[0, m:N-self.M+1+m]
        Y = y
        for d in range(1, D):
            y = np.zeros((N - self.M+1, self.M))
            for m in range(self.M):
                y[:, m] = data[d, m:N-self.M+1+m]
            Y = np.hstack((Y, y))
        # calculate the covariance
        Cemb = np.dot(np.transpose(Y),Y) / (N-self.M+1)
        return Y, Cemb

    @staticmethod
    def _compute_eigenbasis(covar):
        eig_val,eig_vec = np.linalg.eigh(covar)
        indexes = np.argsort(eig_val)[::-1]
        eig_val = np.array([eig_val[i] for i in indexes])
        eig_vec = np.transpose([eig_vec[:, i] for i in indexes])
        return eig_val, eig_vec

    def compute_principal_components(self, data):
        self.N = len(data[0])
        self.D = len(data)

        # compute the covariance
        Y, covar = self._compute_covariance_matrix(data, N, D)
        # compute the eigenbasis
        eig_val, eig_vec = self._compute_eigenbasis(covar)
        # finally the principal components
        PC = np.dot(Y, eig_vec)
        return PC, eig_vec

    def _compute_recontstructed_components(self, PC, eig_vec):
        assert(self.N is not None)
        assert(self.D is not None)
        RC = []
        for d in range(self.D):
            rc = np.zeros((self.N, self.D*self.M))
            for m in range(self.D*self.M):
                buf = np.dot(np.transpose([PC[:, m]]), [np.transpose(eig_vec[d*self.M:(d+1)*self.M, m])])
                buf = np.flipud(buf)
                for n in range(self.N):
                    rc[n, m] = np.mean(np.diag(buf, -(self.N-self.M) + n))
            RC.append(rc)
        return RC

    # def reconstruct_signal(self, pc)