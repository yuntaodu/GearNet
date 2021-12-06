import torch
from models.utils import build_alg_model
from algs.base_model_1 import BaseModel

class GearNet_BaseModel(BaseModel):
    def __init__(self, source_loader, target_loader, device, args):
        if args.direction == 0:
            self.source = source_loader
            self.target = target_loader
        else:
            self.source = target_loader
            self.target = source_loader
        if args.step > 0:
            temp = args.SourceDataset
            args.SourceDataset = args.TargetDataset
            args.TargetDataset = temp

        self.device = device
        self.args = args
        self.aux_model = build_alg_model(args.alg, args, device)

        self.net = build_alg_model(args.alg, args, device)
        self.set_optimizer()

    def cal_KL(self, source_inputs, soft_targets, reduce=True):
        with torch.no_grad():
            y_var, y_softmax_var, d_var = self.aux_model(source_inputs)
        kl_loss1 = self.kl_loss_compute(soft_targets, y_softmax_var, reduce=False)
        kl_loss2 = self.kl_loss_compute(y_softmax_var, soft_targets, reduce=False)
        if reduce:
            kl_loss = torch.mean(kl_loss1 + kl_loss2)
        else:
            kl_loss = kl_loss1 + kl_loss2
        return kl_loss

