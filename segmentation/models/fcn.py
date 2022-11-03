from torch import nn

from torchvision.models.segmentation._utils import _SimpleSegmentationModel

# # JS
# from ...models.gbn import GBN_invariant

__all__ = ["FCN"]


class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),   # classifier.0.weight
            nn.BatchNorm2d(inter_channels),                                     # classifier.1.weight,  classifier.1.bias
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)



class FCNHead_invariant(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),   # classifier.0.weight
            nn.BatchNorm2d(inter_channels, affine=False),   
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)      # 얘를 fix 해야됨 -> classifier.4.weight, classifier.4.bias
        ]

        super(FCNHead_invariant, self).__init__(*layers)