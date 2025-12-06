import os
import modal

app = modal.App("resnet_mini_project")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "pytorch-lightning",
        "torchmetrics",
        "numpy",
        "matplotlib",
    )
    .add_local_file("resnet_miniproject_core.py", "/root/resnet_miniproject_core.py")
)

# @app.function(image=image)
# def ping():
#     return "pong"

@app.function(
    image=image,
    gpu="T4",
    timeout = 60 * 60,
)

def train_on_modal(
    dataset_spec: str,
    mixup_alpha: float,
    run_tag: str,
    max_epochs: int = 20,
    batch_size: int = 128,
    img_size: int = 224,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
):
    """
    Run your existing training helper on a Modal GPU and return checkpoint bytes to be saved locally
    """
    import sys
    sys.path.append("/root")

    # import copied file
    from resnet_miniproject_core import train_and_export_checkpoint

    remote_export_dir = "/root/output"
    os.makedirs(remote_export_dir, exist_ok=True)

    export_path = train_and_export_checkpoint(
        dataset_spec=dataset_spec,
        mixup_alpha=mixup_alpha,
        run_tag=run_tag,
        max_epochs=max_epochs,
        batch_size=batch_size,
        img_size=img_size,
        lr=lr,
        weight_decay=weight_decay,
        export_dir=remote_export_dir,
    )

    # read checkpoint file and send it back as bytes
    with open(export_path, "rb") as f:
        ckpt_bytes = f.read()

    return ckpt_bytes