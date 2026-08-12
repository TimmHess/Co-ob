from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F


def safe_index(lst, index):
    return lst[min(index, len(lst) - 1)]


def get_plugin(strategy, plugin_type):
    for plugin in strategy.plugins:
        if isinstance(plugin, plugin_type):
            return plugin
    return None


def get_optim_and_scheduler(strategy):
    from src.utils.onecyclelr_plugin import OneCycleSchedulerPlugin
    optim_type = type(strategy.optimizer)
    optim_params = strategy.optimizer.defaults
    scheduler_plugin = get_plugin(strategy, OneCycleSchedulerPlugin)  # TODO: This can be abstracted further
    scheduler_type = None
    scheduler_params = None
    if scheduler_plugin is not None:
        scheduler_type = type(scheduler_plugin.scheduler)
        scheduler_params = scheduler_plugin.init_params  # Copy parameters
        
    return {'type': optim_type, 'params': optim_params}, {'type': scheduler_type, 'params': scheduler_params} 


@torch.no_grad()
def prepare_tensordataset(
        model, 
        dataset, 
        batch_size, 
        num_workers, 
        device,
        normalize_features=False
    ):
    x_reprs = []
    ys = []
    ts = []
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    shuffle=False, 
                    num_workers=num_workers, 
                    drop_last=False) 
    
    print("Preparing tensordataset...")
    for _, mbatch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
        
        x = x.to(device)
        y = y.to(device)

        # Get representation from backbone
        x_rep = model(x).detach() # detach() is most likely not necessary here but I want to be sure
        if normalize_features:
            x_rep = F.normalize(x_rep, dim=1)
        x_reprs.append(x_rep.cpu())
        ys.append(y.cpu())
        ts.append(tid)

    x_reprs = torch.concat(x_reprs)
    ys = torch.concat(ys)
    ts = torch.concat(ts)
    return x_reprs, ys, ts


def assert_pil_transform(transform):
    from torchvision.transforms import ToPILImage
    if not isinstance(transform.transforms[0], ToPILImage):
        transform.transforms.insert(0, ToPILImage())
    return transform

def assert_no_pil_transform(transform):
    from torchvision.transforms import ToPILImage
    if isinstance(transform.transforms[0], ToPILImage):
        return transform.transforms[1:]
    return transform

