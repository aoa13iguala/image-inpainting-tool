#!/usr/bin/env python3
"""Remove the leftmost person from a photo and reconstruct the background."""

import argparse
from pathlib import Path


DEFAULT_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def resized_dimensions(image, max_size):
    scale = min(1, max_size / max(image.size))
    width = max(8, round(image.width * scale / 8) * 8)
    height = max(8, round(image.height * scale / 8) * 8)
    return width, height


def detect_leftmost_person(image, confidence):
    import torch
    from PIL import Image
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )
    from torchvision.transforms.functional import to_tensor

    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    detection_image = image.resize(
        resized_dimensions(image, 1024), Image.Resampling.LANCZOS
    )
    with torch.inference_mode():
        result = model([to_tensor(detection_image).to(device)])[0]

    people = [
        index
        for index, label in enumerate(result["labels"].tolist())
        if label == 1 and result["scores"][index].item() >= confidence
    ]
    if not people:
        raise ValueError(
            "No se detectó ninguna persona. Prueba --mask con una máscara manual."
        )

    leftmost = min(people, key=lambda index: result["boxes"][index][0].item())
    person_mask = result["masks"][leftmost, 0].cpu().numpy()
    mask = Image.fromarray((person_mask >= 0.5).astype("uint8") * 255, mode="L")
    return mask.resize(image.size, Image.Resampling.NEAREST)


def prepare_mask(mask, image_size):
    from PIL import Image, ImageFilter

    mask = mask.convert("L").resize(image_size, Image.Resampling.NEAREST)
    if mask.getbbox() is None:
        raise ValueError("La máscara está vacía; debe marcar la zona que se eliminará.")
    padding = max(8, round(max(image_size) * 0.018))
    mask = mask.filter(ImageFilter.MaxFilter(padding * 2 + 1))
    return mask.filter(ImageFilter.GaussianBlur(max(2, padding / 4)))


def inpaint(image, mask, model_id, prompt, steps, guidance_scale, max_size):
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    )
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    size = resized_dimensions(image, max_size)
    work_image = image.resize(size, Image.Resampling.LANCZOS)
    work_mask = mask.resize(size, Image.Resampling.LANCZOS)
    generated = pipeline(
        prompt=prompt,
        image=work_image,
        mask_image=work_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return generated.resize(image.size, Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(
        description="Elimina a la persona situada más a la izquierda y reconstruye el fondo."
    )
    parser.add_argument("image", type=Path, help="Fotografía de entrada")
    parser.add_argument("-o", "--output", type=Path, default=Path("resultado.png"))
    parser.add_argument(
        "--mask",
        type=Path,
        help="Máscara opcional en escala de grises: blanco = eliminar, negro = conservar",
    )
    parser.add_argument(
        "--model", default=DEFAULT_INPAINT_MODEL, help="Modelo de inpainting de Diffusers"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Photorealistic continuation of the existing terrace scene, "
            "matching the original railing, bench, plants, mountain landscape, "
            "natural lighting and camera perspective; no people"
        ),
    )
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"No existe la imagen: {args.image}")
    if args.steps < 1 or args.max_size < 8:
        parser.error("--steps debe ser positivo y --max-size debe ser al menos 8")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence debe estar entre 0 y 1")

    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    if args.mask:
        mask = Image.open(args.mask)
    else:
        mask = detect_leftmost_person(image, args.confidence)
    mask = prepare_mask(mask, image.size)

    mask_path = args.output.with_name(f"{args.output.stem}.mask.png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)

    generated = inpaint(
        image,
        mask,
        args.model,
        args.prompt,
        args.steps,
        args.guidance_scale,
        args.max_size,
    )
    Image.composite(generated, image, mask).save(args.output)
    print(f"Imagen guardada en: {args.output}")
    print(f"Máscara guardada en: {mask_path}")


if __name__ == "__main__":
    main()
