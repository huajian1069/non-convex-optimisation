import numpy as np
import torch
from abc import ABC, abstractmethod

#import seaborn as sns
import matplotlib.pyplot as plt
import plyfile
import skimage.measure
import torch.nn as nn
import torch
import torch.nn.functional as F

class objective_func(ABC):
    @abstractmethod
    def func(self, x):
        pass
    def dfunc(self, x):
        out = self.func(x)
        out.backward()
        return x.grad
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    def visualise1d(self, lim, n):
        ''' 
            lim: the visualisation scope [-lim, lim] in each dimension
            n: the number of points used to interpolate between [-lim, lim]
        '''
        xs = np.linspace(-lim, lim, n)
        fs = []
        for x in xs:
            fs.append(self.func(x))
        plt.plot(xs, fs)
        
class decoder_obj(objective_func):
    def __init__(self, latent_target, decoder):
        self.N_MARCHING_CUBE = 64
        self.l2reg= True
        self.regl2 = 1e-3
        self.iter = 0
        self.quick = True
        
        self.latent_target = latent_target
        self.decoder = decoder
        self.optimum = 0
        self.optimal = latent_target.detach().cpu().numpy()
        
        # Get a mesh representation of the target shape
        self.verts_target, faces_target = create_mesh_optim(
            decoder, latent_target, N=self.N_MARCHING_CUBE, max_batch=int(2 ** 18)
        )
    
        
    def func(self, latent):
        # from latent to xyz
        verts, faces = create_mesh_optim(self.decoder, latent, N=self.N_MARCHING_CUBE, max_batch=int(2 ** 18))
        verts = verts[torch.randperm(verts.shape[0])]
        verts = verts[0:20000, :]
        self.xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32,device=torch.device('cuda:0') )#, ) # For GPU,
       
        # from latent_traget to xyz_target
        verts_target_sample = self.verts_target[torch.randperm(self.verts_target.shape[0])]
        verts_target_sample = verts_target_sample[0:20000, :]
        xyz_target = torch.tensor(verts_target_sample.astype(float), requires_grad = False, dtype=torch.float32, device=torch.device('cuda:0')) # For GPU, add: , device=torch.device('cuda:0'))

        # compare difference
        loss = chamfer_distance(self.xyz_upstream, xyz_target)
        self.last_loss = loss;
        self.last_latent = latent;
        return loss
    
    def dfunc(self, latent):
        if latent.grad is not None:
            latent.grad.detach_()
            latent.grad.zero_()
        
        # step 1
        if self.quick and self.last_latent is not None and torch.all(latent == self.last_latent):
            loss = self.last_loss
        else:
            loss = self.func(latent)
        self.decoder.eval()
        loss.backward()
        dL_dx_i = self.xyz_upstream.grad
        
        # step 2
        # use vertices to compute full backward pass
        xyz = self.xyz_upstream.clone().detach()
        xyz.requires_grad = True
        latent_inputs = latent.expand(xyz.shape[0], -1)
        inputs = torch.cat([latent_inputs, xyz], 1).cuda()      #Add .cuda() if you want to run on GPU
        #first compute normals
        pred_sdf = self.decoder(inputs)
        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)
        normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
                
        # step 3
        # now assemble inflow derivative
        latent.grad.detach_()
        latent.grad.zero_()
        dL_ds_i_fast = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
        loss_backward = torch.sum(dL_ds_i_fast * pred_sdf)
        if self.l2reg and self.iter % 20 == 0 and self.iter > 0:
            self.regl2 = self.regl2/2
        if self.l2reg:
            loss_backward += self.regl2 * torch.mean(latent.pow(2))
        # Backpropagate
        loss_backward.backward()
        
        return latent.grad

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
    
def create_mesh_optim(
    decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None
):

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda() #Add .cuda() if you want to run on GPU

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach().cuda()
            #.cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    verts, faces = convert_sdf_samples_to_mesh(
        sdf_values.data.cuda(), voxel_origin, voxel_size, offset, scale)

    return verts, faces


def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to verts, faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces

def write_verts_faces_to_file(verts, faces, ply_filename_out):

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    
def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
    sdf = decoder(inputs)

    return sdf

def chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    '''
    p1 = p1.unsqueeze(0)
    p2 = p2.unsqueeze(0)

    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    # compute distance tensor
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=2)

    dist1, _ = torch.min(dist, dim = 1)
    dist2, _ = torch.min(dist, dim = 0)

    return torch.mean(dist1) + torch.mean(dist2)

class ackley(objective_func):
    '''
    the period of local minimum along each axis is 1, integer coordinate (1,1), (2,3)... 
    x and y is interchangeable
    global minimum is 0 with arguments x=y=0
    local minimums far away from orgin are 20
    supremum is 20 + e - 1/e = 22.35
    symmetric along x=0, y=0, y=x lines
    disappearing global gradient when far away from optimal
    '''
    def __init__(self, dim=2):
        self.optimum = 0
        self.lim = 5
        self.dim = dim
        self.optimal = np.zeros((self.dim, ))
        self.x = None
        self.out = None
    def func(self, x):

        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = torch.cos(2*np.pi*x).mean() 
        return -20. * torch.exp(arg1) - torch.exp(arg2) + 20. + np.e
    def dfuncR(self, x):
        if torch.norm(x) < 1e-3:
            return torch.zeros((self.dim, ))
        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = torch.cos(2*np.pi*x).mean()
        g = lambda xx: -0.8 * xx / arg1 * torch.exp(arg1) / self.dim + 2 * np.pi * torch.sin(2 * np.pi * xx) * torch.exp(arg2) / self.dim
        return g(x)

    
class bukin(objective_func):
    '''
    non-disappearing gradient
    large gradient and uncontinuous gradient around ridge/local optimal
    optimum: 0
    optimal: (-10, 1)
    '''
    def __init__(self):
        self.optimal = np.array([-10, 1])
        self.optimum = 0
        self.lim = 15
    def func(self, x):
        self.x = x
        self.out = 100 * torch.sqrt(torch.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * torch.abs(x[0] + 10)
        return self.out
    def dfuncR(self, x):
        arg1 = x[1] - 0.01 * x[0]**2
        arg2 = 50 / torch.sqrt(torch.abs(arg1)) * torch.sign(arg1) if arg1 != 0 else 0
        return torch.tensor([- 0.02 * x[0] * arg2 + 0.01 * torch.sign(x[0] + 10), arg2])
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum

class eggholder(objective_func):
    # evaluated domain: 
    def __init__(self):
        self.optimal = np.array([522, 413])
        self.optimum = 0
        self.lim = 550
    def func(self, x):
        if torch.abs(x[0]) > self.lim or torch.abs(x[1]) > self.lim:
            return torch.tensor([2e3], requires_grad=True)
        arg1 = x[0]/2 + (x[1] + 47) 
        arg2 = x[0]   - (x[1] + 47)
        f = lambda xx: torch.sin(torch.sqrt(torch.abs(xx)))
        self.x = x
        self.out = -(x[1] + 47) * f(arg1) - x[0] * f(arg2) + 976.873
        return self.out
    def dfuncR(self, x):
        if torch.abs(x[0]) > self.lim or torch.abs(x[1]) > self.lim:
            return torch.tensor([0, 0])
        arg1 = x[0]/2 + (x[1] + 47) 
        arg2 = x[0]   - (x[1] + 47)
        g = lambda xx: torch.cos(torch.sqrt(torch.abs(xx)))/torch.sqrt(torch.abs(xx))/2*torch.sign(xx)
        f1 = (x[1] + 47) * g(arg1)
        f2 = x[0] * g(arg2)
        return torch.tensor([-f1/2 - torch.sin(torch.sqrt(torch.abs(arg2))) - f2, \
                         -f1 - torch.sin(torch.sqrt(torch.abs(arg1))) + f2]).cuda()
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    
class tuned_ackley(objective_func):
    # evaluated domain: circle with radius 19
    def __init__(self, lim=22, dim=2):
        self.optimum = 0
        self.lim = lim
        self.dim = dim
        self.optimal = np.zeros((self.dim, ))
    def func(self, x):
        '''
        the period of local minimum along each axis is 1, integer coordinate (1,1), (2,3)... 
        x and y is interchangeable
        global minimum is 0 with arguments x=y=0
        symmetric along x=0, y=0, y=x lines
        disappearing global gradient when far away from optimal
        '''
        if torch.norm(x) > self.lim:
            return torch.tensor([5e1], requires_grad=True)
        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = 0.5 * torch.cos(np.pi*x).mean()
        self.x = x
        self.out = -20. * torch.exp(arg1) - 0.1 * arg1**4 * torch.exp(arg2) + 20.
        return self.out
    def dfuncR(self, x):
        if torch.norm(x) < 1e-3:
            return torch.zeros((self.dim,))
        elif torch.norm(x) > self.lim:
            return torch.zeros((self.dim, ))
        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = 0.5 * torch.cos(np.pi*x).mean()
        g = lambda xx: -0.8 * xx / arg1 * torch.exp(arg1) / self.dim + np.pi/20 * arg1**4 * torch.sin(np.pi * xx) * torch.exp(arg2) / self.dim \
                         - 4 * xx/6250 * torch.exp(arg2) * torch.power(x, 2).sum() / self.dim**2
        return g(x)
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    def visualise2d_section(self, pos, dire):
        super().visualise2d_section(pos, dire)
        plt.plot([-25, 25], [15.67, 15.67], label='y=15.67')
        plt.plot([-25, 25], [3.63, 3.66], label='y=3.66')
        plt.plot([12.96, 12.96], [0, 50], label='x=12.96')
        plt.plot([22, 22], [0, 50], label='x=22')
        plt.legend()
