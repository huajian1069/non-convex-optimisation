import os
import sys
from library.cfd_objective import *

def load_pressure_predictor(path):
    model = SplineCNN8Residuals(3)
    model.load_state_dict(torch.load(experiment_directory + "/cfdModel.nn"))
    model = model.to("cuda:0").eval()
    return model

def load_latent_vectors(experiment_directory, checkpoint):
    filename = os.path.join(
        experiment_directory, checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )
    data = torch.load(filename)
    return data["latent_codes"].cuda()

def load_decoder(experiment_directory, checkpoint):
    specs_filename = os.path.join(experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))
    arch = __import__(experiment_directory + "." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(os.path.join(experiment_directory, checkpoint +".pth"))
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    return decoder