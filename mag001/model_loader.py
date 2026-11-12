from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import DeiTConfig, DeiTModel

LOGGER = logging.getLogger(__name__)

REPO_ID = os.getenv(
    "FOOD_CLASSIFIER_REPO_ID", "codealchemist01/food-image-classifier-hybrid"
)
CHECKPOINT_FILENAME = "best_model.pth"
CLASS_MAPPING_FILENAME = "real_class_mapping.json"
FALLBACK_CLASS_NAMES_FILENAME = "food101_class_names.json"

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Network definition (mirrors the author's implementation)
# ---------------------------------------------------------------------------
class CNNBranch(nn.Module):
    """ResNet50 branch producing 7x7 spatial features and a global vector."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ``weights=None`` is the modern equivalent of ``pretrained=False``.
        try:
            resnet = tv_models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:  # very old torchvision
            resnet = tv_models.resnet50(pretrained=pretrained)

        # Drop the final average-pool and fully-connected layers.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_early_layers:
            self._freeze_early_layers()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_proj = nn.Sequential(
            nn.Conv2d(2048, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _freeze_early_layers(self) -> None:
        for index, child in enumerate(self.backbone.children()):
            if index < 6:
                for parameter in child.parameters():
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        spatial_features = self.feature_proj(features)
        global_features = self.global_pool(spatial_features).flatten(1)
        global_features = self.feature_head(global_features)
        return spatial_features, global_features


class ViTBranch(nn.Module):
    """DeiT-Base branch producing patch tokens and a CLS-token vector."""

    def __init__(
        self,
        model_name: str = "facebook/deit-base-distilled-patch16-224",
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        dropout: float = 0.1,
        feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.model_name = model_name

        # The author's Space instantiates DeiTConfig.from_pretrained(...)
        # even when pretrained=False.  The defaults of DeiTConfig match
        # facebook/deit-base-distilled-patch16-224 exactly, so use them to
        # avoid an extra network request.  If you ever change the backbone,
        # update this config (see TODO below).
        #
        # TODO: if a different DeiT variant is used, replace this with
        # ``DeiTConfig.from_pretrained(model_name)``.
        config = DeiTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=IMAGE_SIZE,
            patch_size=16,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.vit = DeiTModel(config)

        if freeze_early_layers:
            self._freeze_early_layers()

        self.hidden_size = self.vit.config.hidden_size
        self.num_patches = (IMAGE_SIZE // 16) ** 2

        self.feature_proj = nn.Sequential(
            nn.Linear(self.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _freeze_early_layers(self) -> None:
        # transformers 4.x calls this ``encoder.layer``; 5.x calls it
        # ``layers``.  Support both so the module can be built anywhere.
        layers = getattr(self.vit, "encoder", None)
        layers = getattr(layers, "layer", None) if layers is not None else None
        if layers is None:
            layers = getattr(self.vit, "layers", [])
        for index, layer in enumerate(layers):
            if index < 8:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vit(pixel_values=x)
        last_hidden_states = getattr(outputs, "last_hidden_state", None)
        if last_hidden_states is None:
            last_hidden_states = outputs[0]

        # Keep the author's indexing: index 0 is the CLS token and the
        # remaining tokens (distillation token + patches) form the spatial
        # sequence.  This matches the checkpoint's training-time behaviour.
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]

        global_features = self.feature_proj(cls_token)
        spatial_features = self.spatial_proj(patch_tokens)
        global_features = self.feature_head(global_features)
        return spatial_features, global_features


class AdaptiveFusionModule(nn.Module):
    """Cross-attention fusion of CNN and ViT features."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
        spatial_size: int = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_size = spatial_size

        self.cnn_to_vit_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vit_to_cnn_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cnn_spatial_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vit_spatial_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adaptive_weight = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cnn_spatial: torch.Tensor,
        cnn_global: torch.Tensor,
        vit_spatial: torch.Tensor,
        vit_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_spatial_seq = cnn_spatial.flatten(2).transpose(1, 2)
        cnn_spatial_proj = self.cnn_spatial_proj(cnn_spatial_seq)
        vit_spatial_proj = self.vit_spatial_proj(vit_spatial)

        cnn_attended, _ = self.cnn_to_vit_attention(
            query=cnn_spatial_proj, key=vit_spatial_proj, value=vit_spatial_proj
        )
        vit_attended, _ = self.vit_to_cnn_attention(
            query=vit_spatial_proj, key=cnn_spatial_proj, value=cnn_spatial_proj
        )

        combined_spatial = torch.cat(
            [cnn_attended + cnn_spatial_proj, vit_attended + vit_spatial_proj], dim=1
        )
        fused_spatial, _ = self.self_attention(
            query=combined_spatial, key=combined_spatial, value=combined_spatial
        )

        global_concat = torch.cat([cnn_global, vit_global], dim=-1)
        fused_global_base = self.global_fusion(global_concat)

        weights = self.adaptive_weight(global_concat)
        cnn_weight = weights[:, 0:1]
        vit_weight = weights[:, 1:2]
        fused_global = (
            cnn_weight * cnn_global + vit_weight * vit_global + fused_global_base
        ) / 2
        fused_global = self.final_proj(fused_global)
        return fused_spatial, fused_global


class HybridFoodClassifier(nn.Module):
    """Hybrid CNN + ViT Food-101 classifier."""

    def __init__(
        self,
        num_classes: int = 101,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = False,
        freeze_early_layers: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.cnn_branch = CNNBranch(
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            dropout=dropout,
            feature_dim=feature_dim,
        )
        self.vit_branch = ViTBranch(
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            dropout=dropout,
            feature_dim=feature_dim,
        )
        self.fusion_module = AdaptiveFusionModule(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.cnn_aux_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.vit_aux_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in [
            self.classifier,
            self.cnn_aux_classifier,
            self.vit_aux_classifier,
        ]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        cnn_spatial, cnn_global = self.cnn_branch(x)
        vit_spatial, vit_global = self.vit_branch(x)
        fused_spatial, fused_global = self.fusion_module(
            cnn_spatial, cnn_global, vit_spatial, vit_global
        )
        logits = self.classifier(fused_global)

        output: Dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            output.update(
                {
                    "cnn_spatial": cnn_spatial,
                    "cnn_global": cnn_global,
                    "vit_spatial": vit_spatial,
                    "vit_global": vit_global,
                    "fused_spatial": fused_spatial,
                    "fused_global": fused_global,
                }
            )
        return output