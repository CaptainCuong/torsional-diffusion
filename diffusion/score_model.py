import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
import numpy as np
from e3nn.nn import BatchNorm
import diffusion.torus as torus
from torch.utils.tensorboard import SummaryWriter

from e3nn.util.jit import compile_mode


# @compile_mode('unsupported')
# class BatchNorm(nn.Module):
#     """Batch normalization for orthonormal representations

#     It normalizes by the norm of the representations.
#     Note that the norm is invariant only for orthonormal representations.
#     Irreducible representations `wigner_D` are orthonormal.

#     Parameters
#     ----------
#     irreps : `o3.Irreps`
#         representation

#     eps : float
#         avoid division by zero when we normalize by the variance

#     momentum : float
#         momentum of the running average

#     affine : bool
#         do we have weight and bias parameters

#     reduce : {'mean', 'max'}
#         method used to reduce

#     instance : bool
#         apply instance norm instead of batch norm
#     """
#     def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, reduce='mean', instance=False, normalization='component'):
#         super().__init__()

#         self.irreps = o3.Irreps(irreps)
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.instance = instance

#         num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
#         num_features = self.irreps.num_irreps

#         if self.instance:
#             self.register_buffer('running_mean', None)
#             self.register_buffer('running_var', None)
#         else:
#             self.register_buffer('running_mean', torch.zeros(num_scalar))
#             self.register_buffer('running_var', torch.ones(num_features))

#         if affine:
#             self.weight = nn.Parameter(torch.ones(num_features))
#             self.bias = nn.Parameter(torch.zeros(num_scalar))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         assert isinstance(reduce, str), "reduce should be passed as a string value"
#         assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
#         self.reduce = reduce

#         assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
#         self.normalization = normalization

#     def __repr__(self):
#         return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

#     def _roll_avg(self, curr, update):
#         return (1 - self.momentum) * curr + self.momentum * update.detach()

#     def forward(self, input):
#         """evaluate

#         Parameters
#         ----------
#         input : `torch.Tensor`
#             tensor of shape ``(batch, ..., irreps.dim)``

#         Returns
#         -------
#         `torch.Tensor`
#             tensor of shape ``(batch, ..., irreps.dim)``
#         """
#         batch, *size, dim = input.shape
#         input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

#         if self.training and not self.instance:
#             new_means = []
#             new_vars = []

#         fields = []
#         ix = 0
#         irm = 0
#         irv = 0
#         iw = 0
#         ib = 0
#         print('*'*10)
#         print(self.irreps)
#         check_state = False
#         check_append = False
#         check_scalar = False
#         for mul, ir in self.irreps:
#             check_state = True
#             d = ir.dim
#             field = input[:, :, ix: ix + mul * d]  # [batch, sample, mul * repr]
#             ix += mul * d

#             # [batch, sample, mul, repr]
#             field = field.reshape(batch, -1, mul, d)
#             if ir.is_scalar():  # scalars
#                 check_scalar = True
#                 if self.training or self.instance:
#                     if self.instance:
#                         field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
#                     else:
#                         field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
#                         check_append = True
#                         new_means.append(
#                             self._roll_avg(self.running_mean[irm:irm + mul], field_mean)
#                         )
#                 else:
#                     field_mean = self.running_mean[irm: irm + mul]
#                 irm += mul

#                 # [batch, sample, mul, repr]
#                 field = field - field_mean.reshape(-1, 1, mul, 1)

#             if self.training or self.instance:
#                 if self.normalization == 'norm':
#                     field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
#                 elif self.normalization == 'component':
#                     field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
#                 else:
#                     raise ValueError("Invalid normalization option {}".format(self.normalization))

#                 if self.reduce == 'mean':
#                     field_norm = field_norm.mean(1)  # [batch, mul]
#                 elif self.reduce == 'max':
#                     field_norm = field_norm.max(1).values  # [batch, mul]
#                 else:
#                     raise ValueError("Invalid reduce option {}".format(self.reduce))

#                 if not self.instance:
#                     field_norm = field_norm.mean(0)  # [mul]
#                     new_vars.append(self._roll_avg(self.running_var[irv: irv + mul], field_norm))
#             else:
#                 field_norm = self.running_var[irv: irv + mul]
#             irv += mul

#             field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

#             if self.affine:
#                 weight = self.weight[iw: iw + mul]  # [mul]
#                 iw += mul

#                 field_norm = field_norm * weight  # [(batch,) mul]

#             field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

#             if self.affine and ir.is_scalar():  # scalars
#                 bias = self.bias[ib: ib + mul]  # [mul]
#                 ib += mul
#                 field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

#             fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]
#         print('In loop: ', check_state)
#         print('Scalar: ', check_scalar)
#         print('Already append: ', check_append)
#         if ix != dim:
#             fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
#             msg = fmt.format(dim, ix)
#             raise AssertionError(msg)

#         if self.training and not self.instance:
#             assert irm == self.running_mean.numel()
#             assert irv == self.running_var.size(0)
#         if self.affine:
#             assert iw == self.weight.size(0)
#             assert ib == self.bias.numel()
#         print('*'*10)
#         if self.training and not self.instance:
#             torch.cat(new_means, out=self.running_mean)
#             torch.cat(new_vars, out=self.running_var)

#         output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
#         return output.reshape(batch, *size, dim)

class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, in_node_features=74, in_edge_features=4, sigma_embed_dim=32, sigma_min=0.01 * np.pi,
                 sigma_max=np.pi, sh_lmax=2, ns=32, nv=8, num_conv_layers=4, max_radius=5, radius_embed_dim=50,
                 scale_by_sigma=True, use_second_order_repr=True, batch_norm=True, residual=True
                 ):
        super(TensorProductScoreModel, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma

        self.node_embedding = nn.Sequential(
            nn.Linear(in_node_features + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_features + sigma_embed_dim + radius_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )
        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)
        conv_layers = []

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.final_edge_embedding = nn.Sequential(
            nn.Linear(radius_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )

        self.final_tp = o3.FullTensorProduct(self.sh_irreps, "2e")

        self.bond_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps,
            sh_irreps=self.final_tp.irreps_out,
            out_irreps=f'{ns}x0o',
            n_edge_features=3 * ns,
            residual=False,
            batch_norm=None
        )

        self.final_linear = nn.Sequential(
            nn.Linear(ns, ns, bias=False),
            nn.Tanh(),
            nn.Linear(ns, 1, bias=False)
        )

    def forward(self, data):
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
        src, dst = edge_index

        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.conv_layers:
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')

        bonds, edge_index, edge_attr, edge_sh = self.build_bond_conv_graph(data, node_attr)
        #######################
        bond_vec = data.pos[bonds[1]] - data.pos[bonds[0]]
        bond_attr = node_attr[bonds[0]] + node_attr[bonds[1]]
        # Y(r_ab)
        bonds_sh = o3.spherical_harmonics("2e", bond_vec, normalize=True, normalization='component')
        # edge_sh.shape: torch.Size([2017, 9])
        edge_sh = self.final_tp(edge_sh, bonds_sh[edge_index[0]]) # Psi
        # bond_vec.shape: torch.Size([107, 3])
        # bonds_sh.shape: torch.Size([107, 5]) -> bonds_sh[edge_index[0]]:
        # edge_index.shape: torch.Size([2, 2017])
        # edge_sh.shape: torch.Size([2017, 45]) = ([2017, 9]) * ([2017, 5])
        edge_attr = torch.cat([edge_attr, node_attr[edge_index[1], :self.ns], bond_attr[edge_index[0], :self.ns]], -1)
        out = self.bond_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=data.edge_mask.sum(), reduce='mean')

        out = self.final_linear(out)
        #######################
        data.edge_pred = out.squeeze()
        data.edge_sigma = data.node_sigma[data.edge_index[0]][data.edge_mask]
        # data.node_sigma.shape: (445,)
        # data.node_sigma[data.edge_index[0]].shape: (838,) because data.edge_index[0] has repetitive elments
        if self.scale_by_sigma:
            score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
            score_norm = torch.tensor(score_norm, device=data.x.device)
            data.edge_pred = data.edge_pred * torch.sqrt(score_norm)
        # bonds.shape: torch.Size([2, 107])
        # data.edge_pred.shape: torch.Size([107])
        return data

    def build_bond_conv_graph(self, data, node_attr):

        bonds = data.edge_index[:, data.edge_mask].long()
        bond_pos = (data.pos[bonds[0]] + data.pos[bonds[1]]) / 2
        bond_batch = data.batch[bonds[0]]
        edge_index = radius(data.pos, bond_pos, self.max_radius, batch_x=data.batch, batch_y=bond_batch)

        edge_vec = data.pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

    def build_conv_graph(self, data):
        radius_edges = radius_graph(data.pos, self.max_radius, data.batch)
        edge_index = torch.cat([data.edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data.edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_edge_features, device=data.x.device)
        ], 0)
        '''
        Add edges + attributes within radius
        '''
        node_sigma = torch.log(data.node_sigma / self.sigma_min) / np.log(self.sigma_max / self.sigma_min) * 10000
        node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim)
        edge_sigma_emb = node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data.x, node_sigma_emb], 1)
        src, dst = edge_index
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1) # edge_attr || edge_sigma_emb || edge_length_emb

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)
        
    def forward(self, dist):
        '''
        a = (stop-start)/num_gaussians
        Return: [exp(-0.5/a^2*(dist-start)^2), exp(-0.5/a^2*(dist-start+a)^2),..., exp(-0.5/a^2*(dist-stop)^2)]
        '''
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    # writer = SummaryWriter('runs/timesteps')
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb) # [embedding_dim // 2]
    # [writer.add_scalar(f'Timesteps/array', x, ind) for ind, x in enumerate(emb)]
    emb = timesteps.float()[:, None] * emb[None, :] # [timesteps.shape[0], embedding_dim//2]
    # [writer.add_scalar(f'Timesteps/timesteps', x, ind) for ind, x in enumerate(timesteps)]
    # [writer.add_scalar(f'Timesteps/float{0}', x, ind) for ind, x in enumerate(emb[0])]
    # [writer.add_scalar(f'Timesteps/float{emb.shape[0]}', x, ind) for ind, x in enumerate(emb[-1])] 
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # [timesteps.shape[0], embedding_dim//2*2]
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant') # add a padding at the end of the last dimension
    # [writer.add_scalar(f'Timesteps/final{0}', x, ind) for ind, x in enumerate(emb[0])]
    # [writer.add_scalar(f'Timesteps/final{emb.shape[0]}', x, ind) for ind, x in enumerate(emb[-1])] 
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
