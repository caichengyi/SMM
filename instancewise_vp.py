from functools import partial
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import sys

sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from labelmapping import generate_label_mapping_by_frequency, label_mapping_base
from instance_model import InstancewiseVisualPrompt
from cfg import *

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18", "resnet50", "ViT_B32"], default="resnet18")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset',
                   choices=["cifar10", "cifar100", "gtsrb", "svhn"], default="cifar10")
    p.add_argument('--patch_size', type=int, default=8)
    p.add_argument('--attribute_channels', type=int, default=3)
    p.add_argument('--mapping_method', type=str, default='ilm')

    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    attribute_layers, epochs, lr, attr_lr, attr_gamma = get_config(args.network)
    save_path = os.path.join(results_path, args.dataset + args.network + args.mapping_method + str(args.seed) + str(args.attribute_channels) + str(attribute_layers) + str(args.patch_size))

    if args.network == "ViT_B32":
        imgsize = 384
    else:
        imgsize = 224

    # Data
    train_preprocess = transforms.Compose([
        transforms.Resize((imgsize + 32, imgsize + 32)),
        transforms.RandomCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path, preprocess=train_preprocess,
                                                     test_process=test_preprocess)

    # Network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif args.network == "ViT_B32":
        from pytorch_pretrained_vit import ViT
        model_name = 'B_32_imagenet1k'
        network = ViT(model_name, pretrained=True).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    network.requires_grad_(False)
    network.eval()

    # Visual Prompt
    visual_prompt = InstancewiseVisualPrompt(imgsize, attribute_layers, args.patch_size, args.attribute_channels).to(device)

    # optimizers
    optimizer = torch.optim.Adam([{'params': visual_prompt.program, 'lr': lr}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(0.5 * epochs), int(0.72 * epochs)],
                                                         gamma=0.1)
    optimizer_att = torch.optim.Adam([{'params': visual_prompt.priority.parameters(), 'lr': attr_lr}])
    scheduler_att = torch.optim.lr_scheduler.MultiStepLR(optimizer_att,
                                                     milestones=[int(0.5 * epochs), int(0.72 * epochs)],
                                                     gamma=attr_gamma)

    # Make dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(save_path)

    # label_mapping method
    if args.mapping_method == 'rlm':
        mapping_sequence = torch.randperm(1000)[:len(class_names)]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    elif args.mapping_method == 'flm':
        mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, network, loaders['train'])
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)

    # Train
    best_acc = 0.
    scaler = GradScaler()
    for epoch in range(epochs):
        if args.mapping_method == 'ilm':
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, network, loaders['train'])
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Epo {epoch}", ncols=100)
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            pbar.set_description_str(f"Epo {epoch}", refresh=True)
            optimizer.zero_grad()
            optimizer_att.zero_grad()
            with autocast():
                fx = label_mapping(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_att)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100 * true_num / total_num:.2f}%")

        scheduler.step()
        scheduler_att.step()
        logger.add_scalar("train/acc", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        ys = []
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_sequence": mapping_sequence,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))
