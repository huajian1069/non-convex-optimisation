#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#  python optim.py -s example1/synth_test.json -e example1
import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws

import pdb

from library.optimiser import *
from library.objective_function import *
#from library.post_analysis import *
from library.experiments import *
from deep_sdf.mesh import create_mesh_optim


def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

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

class argms:
    def __init__(self):
        self.experiment_directory = "example1"
        self.checkpoint = "latest"
        self.iterations = 100
        self.split_filename = "example1/synth_test.json"
        self.logfile = None
        self.debug = False
        self.quiet = False
args = argms()

def getLatentSourceAndTarget(args, source_id, target_id):
    # pick initialization and samples
    # Load collection of all latent codes
    all_codes_path = os.path.join(
        args.experiment_directory,
        ws.latent_codes_subdir,
        'latest.pth')
    all_codes = torch.load(all_codes_path)['latent_codes']['weight']
    ## sphere
    source_id = 999 # zywvjkvz2492e6xpq4hd1jzy2r9lht        # This will be the source shape (ie starting point)
    latent = all_codes[source_id].unsqueeze(0).detach().cuda()   #Add .cuda() if you want to run on GPU
    latent.requires_grad = True

    # This is be the target shape (ie objective)
    latent_target = all_codes[target_id].unsqueeze(0).detach().cuda()   #Add .cuda() if you want to run on GPU
    return latent, latent_target

def constructDecoder(args):
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    specs = json.load(open(specs_filename))
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    # Load decoder: this is our black box function
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        ),
       # map_location=torch.device('cpu') # Remove this if you want to run on GPU
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    # Optionally: put decoder on GPU
    decoder = decoder.module.cuda()
    return decoder

class decoder_obj(objective_func):
    def __init__(self, latent_target, decoder):
        self.N_MARCHING_CUBE = 64
        self.l2reg= True
        self.regl2 = 1e-3
        self.iter = 0
        self.quick = False
        
        self.latent_target = latent_target
        self.decoder = decoder
        self.optimum = 0
        self.optimal = latent_target
        
        # Get a mesh representation of the target shape
        self.verts_target, faces_target = create_mesh_optim(
            decoder, latent_target, N=self.N_MARCHING_CUBE, max_batch=int(2 ** 18)
        )
    
        
    def func(self, latent):
        # from latent to xyz
        verts, faces = deep_sdf.mesh.create_mesh_optim(self.decoder, latent, N=self.N_MARCHING_CUBE, max_batch=int(2 ** 18))
        verts = verts[torch.randperm(verts.shape[0])]
        verts = verts[0:20000, :]
        self.xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0') )#, ) # For GPU,
       
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
        if self.quick and torch.norm(latent - self.last_latent):
            loss = self.last_loss
        else:
            loss = self.func(latent)
        decoder.eval()
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
        if l2reg and self.iter % 20 == 0 and self.iter > 0:
            self.regl2 = self.regl2/2
        if l2reg:
            loss_backward += self.regl2 * torch.mean(latent.pow(2))
        # Backpropagate
        loss_backward.backward()
        
        return latent.grad
        
        
if __name__ == "__main__":
    
    torch.manual_seed(0)
    # 0 Initialization
    N_MARCHING_CUBE = 64
    lr= 8e-3
    l2reg= True
    regl2 = 1e-3
    decreased_by = 1.5
    adjust_lr_every = 50
    
    # 1 prepare data
    ## sphere
    source_id = 999 # zywvjkvz2492e6xpq4hd1jzy2r9lht        # This will be the source shape (ie starting point)
    ## torus
    target_id = 2 # 0bucd9ryckhaqtqvbiagilujeqzek4  
    latent, latent_target = getLatentSourceAndTarget(args, source_id, target_id)
    
    # 2 prepare model
    decoder = constructDecoder(args)
    # 3 prepare optimiser
    optimizer = torch.optim.Adam([latent], lr=lr)

    losses = []
    lambdas = []
    

    objectiveDe = decoder_obj(latent_target, decoder)

    # Use Adam optimizer, with source as starting point, and a loss defined on meshes
    # latent is the input of our function
    print("Starting optimization:")
    for e in range(int(args.iterations)):
        print("latent: ", latent.detach().numpy())
        
        loss = objectiveDe.func(latent)
        losses.append(loss.detach().cpu().numpy()) 
        print("loss: ", loss.detach().numpy())
        
        grad = objectiveDe.dfunc(latent)
        print("latent grad: ", grad.detach().numpy())

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)
        optimizer.step()
        print(e, "th iteration\n")

