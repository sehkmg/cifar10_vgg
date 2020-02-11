import torch.nn as nn

vgg11_arch = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def make_conv_layer(arch):
    in_channels = 3
    module_list = []

    for layer_info in arch:
        if layer_info != 'M':
            out_channels = layer_info
            module_list += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()]
            in_channels = out_channels
        else:
            module_list += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]

    return nn.Sequential(*module_list)

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = make_conv_layer(vgg11_arch)

        self.fc_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        conv_out = self.conv_layer(x)
        conv_out = conv_out.flatten(1)

        fc_out = self.fc_layer(conv_out)

        return fc_out