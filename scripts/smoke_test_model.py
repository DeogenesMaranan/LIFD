import sys
sys.path.append('.')
import torch
from model.hybrid_forgery_detector import HybridForgeryConfig, HybridForgeryDetector
from losses.segmentation_losses import CombinedSegmentationLoss, LossConfig

print('PyTorch version:', torch.__version__)

def run():
    cfg = HybridForgeryConfig(pretrained_backbones=False)
    model = HybridForgeryDetector(cfg)
    model.eval()

    x = torch.rand(1, 3, cfg.backbone_input_size if isinstance(cfg.backbone_input_size, int) else cfg.backbone_input_size[0],
                   cfg.backbone_input_size if isinstance(cfg.backbone_input_size, int) else cfg.backbone_input_size[1])
    print('Input shape:', x.shape)

    with torch.no_grad():
        logits = model(x)
    print('Logits shape:', logits.shape)

    # predict_mask without threshold should return logits
    pm = model.predict_mask(x)
    print('predict_mask (no threshold) shape:', pm.shape)

    # predict_mask with threshold should return binary mask
    pm_thr = model.predict_mask(x, threshold=0.5)
    print('predict_mask (with threshold) unique values:', torch.unique(pm_thr))

    # compute loss
    loss_fn = CombinedSegmentationLoss(LossConfig())
    targets = torch.rand_like(logits)
    loss, terms = loss_fn(logits, targets)
    print('Loss:', loss.item())
    print('Loss terms keys:', list(terms.keys()))

if __name__ == '__main__':
    run()
