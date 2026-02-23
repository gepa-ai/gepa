# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code modified from here
https://github.com/albanie/collaborative-experts/blob/master/model/net_vlad.py
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Model(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK

batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
  return [torch.rand(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]
