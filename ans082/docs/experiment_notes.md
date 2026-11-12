# Experiment Notes

All experiments use the same data split (`splits/split_v1.csv`, seed 42), the same 8 epochs,
batch size 32, AdamW with weight decay 0.01, cosine learning-rate decay, fp16 and seed 42,
and run on a Google Colab Tesla T4. Every setting is stored in a config file in `configs/`.
The test split and the real photos were not used for any training or model-selection decision.

## Experiments

| ID | Model | What is trained | Learning rate | Augmentation | Config |
|---|---|---|---|---|---|
| E1 | ResNet-50 (ImageNet) | all layers | 1e-4 | standard | `resnet50.yaml` |
| E2 | ViT-B/16 (google/vit-base-patch16-224) | classification head only | 1e-3 (10% warm-up) | standard | `vit_head_only.yaml` |
| E3 | ViT-B/16 | all layers | 2e-5 (10% warm-up) | standard | `vit_full.yaml` |
| E4 | ViT-B/16 | all layers | 2e-5 (10% warm-up) | strong | `vit_full_strong_aug.yaml` |

Standard augmentation: random resized crop (scale 0.7 to 1.0), horizontal flip, rotation up to
15 degrees, light colour jitter. Strong augmentation (E4) adds perspective changes, blur,
occasional grayscale, stronger colour jitter and random erasing, to imitate real phone photos.

## Validation results

| ID | Trainable parameters | Best epoch | Val accuracy | Val macro-F1 | Training time | Time per epoch |
|---|---|---|---|---|---|---|
| E1 | 23.5 M | 8 | 0.9755 | 0.9682 | 15.8 min | 119 s |
| E2 | 7,690 | 7 | 0.9780 | 0.9710 | 17.5 min | 132 s |
| E3 | 85.8 M | 7 | 0.9858 | 0.9825 | 20.9 min | 156 s |
| E4 | 85.8 M | 5 | 0.9841 | 0.9777 | 25.7 min | 193 s |

## Observations
- Each run is a single seed, and one percentage point is about 23 of the 2,323 validation
  images, so differences below about half a point should not be over-interpreted.
- Training only the 7,690-parameter head of the ViT (E2) already matched the fully fine-tuned
  ResNet-50 (E1) on validation, which shows how strong the pretrained ViT features are.
- Full fine-tuning (E3) gave the highest validation accuracy and macro-F1.
- Stronger augmentation (E4) did not improve validation scores (0.9841 and 0.9777 against
  0.9858 and 0.9825). This is expected, because validation images come from the same
  distribution as the training images. Whether it helps on real phone photos is tested in the
  final evaluation, without using that result to choose the final model.
- Stronger augmentation made each epoch about 23% slower (CPU-bound image processing).
- The ViT run time per epoch is only slightly higher than the ResNet-50's because data loading
  (two CPU workers) is a large part of every epoch.

## Model selection
The final model is chosen by validation macro-F1 only: **E3 (ViT-B/16, full fine-tuning)**,
validation macro-F1 0.9825, epoch 7 checkpoint. Test and real-photo results are reported for
all experiments, but they did not influence this choice.
