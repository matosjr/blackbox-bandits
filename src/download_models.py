import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
vgg16bn = models.vgg16_bn(pretrained=True)

vgg19 = models.vgg19(pretrained=True)
vgg19_bn = models.vgg19_bn(pretrained=True)

googlenet = models.googlenet(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
densenet = models.densenet161(pretrained=True)

# googlenet = models.googlenet(pretrained=True)
# inception = models.inception_v3(pretrained=True)