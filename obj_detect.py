#obj_detect
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator

from head_detection.models.head_detect import create_backbone
from head_detection.models.fast_rcnn import FasterRCNN

from tqdm import tqdm
from matplotlib.pyplot import imread
from glob import glob
import os.path as osp


class HeadHunter(FasterRCNN):
    
    def __init__(self, net_cfg, det_cfg, im_shape, im_dir, custom_anchor=False):

        self.__device = torch.device("cuda")

        # Init and load model
        kwargs = {}
        kwargs['min_size'] = None
        kwargs['max_size'] = None
        dset_mean_std = [[117, 110, 105], [67.10, 65.45, 66.23]]
        kwargs['image_mean'] = [i/255. for i in dset_mean_std[0]]
        kwargs['image_std'] = [i/255. for i in dset_mean_std[1]]
        kwargs['box_score_thresh'] = det_cfg['confidence_threshold']
        kwargs['box_nms_thresh'] = det_cfg['nms_threshold']
        kwargs['box_detections_per_img'] = 300
        kwargs['num_classes'] = 2
        kwargs['cfg'] = net_cfg
        backbone = create_backbone(cfg=net_cfg, context=det_cfg['context'],
                                   use_deform=det_cfg['use_deform'],
                                   default_filter=False,
                                   )
        kwargs['backbone'] = backbone

        if det_cfg['median_anchor']:
            if det_cfg['benchmark'] == 'CHuman':
                from head_detection.data.anchors import ch_anchors as anchors
            elif det_cfg['benchmark'] == 'SHead':
                from head_detection.data.anchors import sh_anchors as anchors
            elif det_cfg['benchmark'] == 'Combined':
                from head_detection.data.anchors import combined_anchors as anchors
            else:
                raise ValueError("Unsupported benchmark")

            anchor_sizes = anchors['anchor_sizes']
            aspect_ratios = anchors['aspect_ratios']
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            kwargs['rpn_anchor_generator'] = rpn_anchor_generator

        super(HeadHunter, self).__init__(**kwargs)

        restore_network(self, det_cfg['trained_model'])
        self.to(self.__device)
        self.eval()
        print("Initialized Detector in testing mode")

    def __preprocess_im(self, im):
        """
        from (H, W, C) -> (1, C, H, W)
        """
        from albumentations.pytorch import ToTensor
        # Preprocess im
        if not isinstance(im, np.ndarray):
            raise ValueError("Wrong image type.")

        transf = ToTensor()
        torched_im = transf(image=im)['image'].to(self.__device)
        # return 
        return torch.unsqueeze(torched_im, 0)

    @torch.no_grad()
    def predict_box(self, im):
        # MP related configuration
        torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")

        torched_im = self.__preprocess_im(im)
        
        outputs = self(torched_im)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']
        
        # move the values back to original dimension
        return boxes, scores

    @torch.no_grad()
    def regress_boxes(self, images, boxes):
        images = self.__preprocess_im(images)
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32).to(self.__device)

        targets = None
        original_image_size = images.shape[-2:]
        images, targets = self.transform(images, targets)
        transformed_image_size = images.image_sizes[0]
        self.features = self.backbone(images.tensors)

        # proposals, proposal_losses = self.rpn(images, features, targets)
        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(boxes, original_image_size, transformed_image_size)
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(
            self.features, proposals, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, transformed_image_size, original_image_size)
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes.cpu(), pred_scores.cpu().numpy()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        self.features = self.backbone(images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
        proposals, proposal_losses = self.rpn(images, self.features, targets)
        detections, detector_losses = self.roi_heads(self.features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections


def restore_network(net, pretrained_path):
    print('Loading resume network...')
    state_dict = torch.load(pretrained_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net

