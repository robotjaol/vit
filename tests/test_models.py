import pytest
import torch

from vit_pytorch import SimpleViT, ViT


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_vit_output_shape(pool):
    model = ViT(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        pool=pool,
        head_dim=8,
    )
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_simple_vit_output_shape():
    model = SimpleViT(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        head_dim=8,
    )
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_invalid_image_size_is_rejected():
    with pytest.raises(AssertionError):
        ViT(
            image_size=30,
            patch_size=8,
            num_classes=10,
            dim=32,
            depth=1,
            num_heads=4,
            mlp_dim=64,
            head_dim=8,
        )
