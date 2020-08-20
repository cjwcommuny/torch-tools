from collections import OrderedDict

import torch
import torchvision.transforms.functional as torchvision_fn
from torch import Tensor
from torch import nn
from torch.jit.annotations import List, Dict, Tuple
from torchvision.models import resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.resnet import Bottleneck, ResNet


def resnet50_from_checkpoint(file_like_obj):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.load(file_like_obj))
    return model


class ResNetFeatureExtractor(nn.Module):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DIM_OUTPUT = 2048

    def __init__(self, checkpoint=None):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True) if checkpoint is None \
            else resnet50_from_checkpoint(checkpoint)
        self.model = nn.Sequential(OrderedDict([
            ("conv1", resnet.conv1),
            ("bn1", resnet.bn1),
            ("relu", resnet.relu),
            ("maxpool", resnet.maxpool),
            ("layer1", resnet.layer1),
            ("layer2", resnet.layer2),
            ("layer3", resnet.layer3),
            ("layer4", resnet.layer4),
            ("avgpool", resnet.avgpool)
        ]))

    def forward(self, x: Tensor):
        """
        :param x: shape=(N, C=3, H>=224, W>=224), range in [0,1], not normalized
        :return: shape=(N, DIM_OUTPUT)
        """
        x = [torchvision_fn.normalize(image, self.IMAGENET_MEAN, self.IMAGENET_STD)
             for image in torch.unbind(x)]
        x = torch.stack(x)
        x = self.model(x)
        x = torch.flatten(x, 1)
        return x


class RcnnFeatureExtractor(nn.Module):
    """
    WARNING: only test on:
        - torch.__version__ == '1.3.1'
        - torchvision.__version__ == '0.4.2'
    """
    def __init__(self, Rcnn, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100):
        super().__init__()
        assert Rcnn in {fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn}

        self.rcnn: GeneralizedRCNN = Rcnn(
            pretrained=True,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img
        )
        self.eval()


    def forward(self, images: List[Tensor], return_image_feature=False) -> List[Dict]:
        """
        :param images: shape=(C, H, W),
            element in [0,1], (by PIL.Image -> torchvision.transforms.functional.to_tensor),
            no need to normalized
        :param return_image_feature
        :return:
            dict.keys:
                'boxes': coordinates of boxes, (x1, y1, x2, y2)
                'box_features':
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        #
        targets = None
        images, _ = self.rcnn.transform(images, targets)
        features = self.rcnn.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rcnn.rpn(images, features, targets)
        detections, detector_losses = self.rcnn.roi_heads(
            features, proposals, images.image_sizes, targets)
        detections = self.rcnn.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        #
        boxes_in_original_image = [d["boxes"] for d in detections]
        if return_image_feature:
            whole_image_boxes = [
                torch.tensor([[0,0,w,h]]).type_as(boxes_in_original_image[0])
                for h, w in original_image_sizes]
            boxes_in_original_image = [
                torch.cat((whole_image, b), 0)
                for whole_image, b in zip(whole_image_boxes, boxes_in_original_image)]
        boxes_in_internal_image = [
            self.resize_boxes(boxes, original_size, internal_size)
            for boxes, original_size, internal_size
            in zip(boxes_in_original_image, original_image_sizes, images.image_sizes)
        ]
        final_box_features = self.proposals_to_features(
            boxes_in_internal_image, features, images.image_sizes)
        for i in range(len(final_box_features)):
            detections[i]["box_features"] = final_box_features[i]
        return detections


    def proposals_to_features(
            self,
            proposals: List[Tensor],
            image_features,
            image_shapes: List[Tuple[int, int]]):
        """
        NOTE: image here means internal form of RCNN, e.g. origianl image shape is (480, 640), the
        internal image shape might be (800, 1066)

        :param proposals: [Tensor[num_boxes, 4]], NOTE: coordinates are measured in internal image shape
        :param image_features: type=OrderedDict[str, Tensor]
        :param image_shapes: [(h, w)]
        :return:
            box_features: shape=[Tensor[num_boxes, dim_feature]]
        """
        box_features = self.rcnn.roi_heads.box_roi_pool(image_features, proposals, image_shapes)
        box_features = self.rcnn.roi_heads.box_head(box_features)
        box_counts = [p.shape[0] for p in proposals]
        box_features = box_features.split(box_counts)
        return box_features


    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        """
        resize box from original_size to new_size
        :param boxes: shape=(num_boxes, 4)
        :param original_size: (h, w)
        :param new_size: (h,w)
        :return:
            new_boxes: shape=(num_boxes, 4)
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)