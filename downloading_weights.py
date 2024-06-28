import os
import wget
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def download_models(model_dir: str = 'pretrained_weights'):
    os.makedirs(model_dir, exist_ok=True)

    urls = [
        'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        # Hugging Face model identifiers
        ('yzd-v/DWPose', 'dw-ll_ucoco_384.pth'),
        ('TMElyralab/MusePose', 'MusePose/denoising_unet.pth'),
        ('TMElyralab/MusePose', 'MusePose/motion_module.pth'),
        ('TMElyralab/MusePose', 'MusePose/pose_guider.pth'),
        ('TMElyralab/MusePose', 'MusePose/reference_unet.pth'),
        ('lambdalabs/sd-image-variations-diffusers', 'unet/diffusion_pytorch_model.bin'),
        ('lambdalabs/sd-image-variations-diffusers', 'image_encoder/pytorch_model.bin'),
        ('stabilityai/sd-vae-ft-mse', 'diffusion_pytorch_model.bin')
    ]

    paths = [
        'dwpose', 'dwpose', '', '', '', '', 
        'sd-image-variations-diffusers', '', 'sd-vae-ft-mse'
    ]

    for path in paths:
        dir = os.path.join(model_dir, path)
        os.makedirs(dir, exist_ok=True)

    for url, path in tqdm(zip(urls, paths)):
        if isinstance(url, tuple):
            repo_id, filename = url
            full_file_path = os.path.join(model_dir, path, filename)
            if not os.path.exists(full_file_path):
                print(f"Model '{filename}' does not exist. Downloading to '{full_file_path}'..")
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.join(model_dir, path))
        else:
            filename = os.path.basename(url)
            if filename == "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth":
                filename = "yolox_l_8x8_300e_coco.pth"
            full_file_path = os.path.join(model_dir, path, filename)
            if not os.path.exists(full_file_path):
                print(f"Model '{filename}' does not exist. Downloading to '{full_file_path}'..")
                wget.download(url, full_file_path)

    config_urls = [
        ('lambdalabs/sd-image-variations-diffusers', 'unet/config.json'),
        ('lambdalabs/sd-image-variations-diffusers', 'image_encoder/config.json'),
        ('stabilityai/sd-vae-ft-mse', 'config.json')
    ]

    config_paths = ['sd-image-variations-diffusers', '', 'sd-vae-ft-mse']

    for url, path in tqdm(zip(config_urls, config_paths)):
        repo_id, filename = url
        full_file_path = os.path.join(model_dir, path, filename)
        if not os.path.exists(full_file_path):
            print(f"Config '{filename}' does not exist. Downloading to '{full_file_path}'..")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.join(model_dir, path))

# 运行下载函数
download_models()
