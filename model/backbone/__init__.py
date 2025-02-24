from model.backbone import resnet

def build_backbone(back_bone):
    # change the resnet101 pretrained to False
    if back_bone == "resnet101":
        return resnet.ResNet101(pretrained=False)
