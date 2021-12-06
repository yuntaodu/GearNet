import torch

def build_alg_model(alg, args, device):
    if alg == "gearnet_coteaching":
        from models.classifier import Classifier
        net = Classifier(args, bottleneck=args.bottleneck)
    elif alg == 'gearnet_dann' or alg == 'gearnet_tcl':
        from models.dann import DANN_net
        net = DANN_net(args, bottleneck=args.bottleneck, num_hidden=args.num_hidden)

    net.to(device)
    if args.gpu is not None and len(args.gpu) > 1:
        gpu_list = [int(s) for s in args.gpu.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net