import torch


class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate(self, x, class_idx):
        x.requires_grad_(True)

        features = self.model.forward_features(x)
        features.retain_grad()

        pooled = self.model.backbone.avgpool(features)
        logits = self.model.head(pooled)

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward()

        grads = features.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1)

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam
