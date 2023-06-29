import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()
import torch
torch.save(model.state_dict(),'pretrained/vit_base_patch16_224.pt')
# import torchvision
# cifar100 = torchvision.datasets.CIFAR100('./data',train=True,transform=None,download=True)