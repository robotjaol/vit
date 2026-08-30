# Vision Transformer Study Project

Readable PyTorch implementations of the original Vision Transformer and a simplified mean-pooled variant, plus a notebook that explains patch embeddings, attention, positional encoding, training, and evaluation.

> **Project status:** educational implementation with a small API; suitable for study and experimentation, not a drop-in production model package.

## What this repository contains

- ViT with a learnable class token and positional embeddings.
- SimpleViT with mean pooling and fixed two-dimensional sinusoidal positions.
- Pre-LayerNorm attention and feed-forward blocks.
- A CIFAR-10 tutorial notebook with training and visualisation.
- Shape-oriented tests and package metadata.

## Quick start


~~~bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
jupyter lab exam/vit_tutorial.ipynb
~~~

## Engineering notes

- Input height and width must be divisible by the patch dimensions.
- For SimpleViT, the model dimension must be divisible by four to construct the 2D sinusoidal embedding.
- The implementations return logits with shape batch × classes; setting num_classes to zero on ViT returns token features.
- Benchmark memory and accuracy against the exact dataset, resolution, seed, and augmentation policy.

## Repository map

| Path | Purpose |
| --- | --- |
| vit_pytorch/vit.py | Class-token Vision Transformer. |
| vit_pytorch/simple_vit.py | Mean-pooled SimpleViT. |
| exam/vit_tutorial.ipynb | Step-by-step learning notebook. |
| tests/ | Fast API and shape checks. |
| pyproject.toml | Installable package metadata. |

## Safety and limitations

Model output is statistical and can fail under distribution shift. Do not use this educational implementation for high-stakes decisions without dataset governance, bias analysis, calibrated evaluation, monitoring, and human oversight.

## Contributing

Open an issue before a large change. Keep changes focused, document assumptions, and include a reproducible verification step.

## License

A repository-wide open-source license has not been declared. Obtain permission before redistributing material.
