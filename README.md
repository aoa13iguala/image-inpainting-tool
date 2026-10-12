# image-inpainting-tool

Herramienta de línea de comandos para eliminar a la persona situada más a la
izquierda de una fotografía y reconstruir el fondo con un modelo de difusión
XL de inpainting. La composición final conserva exactamente los píxeles
originales que están fuera de la máscara.

## Requisitos

- Python 3.10 o posterior
- Un entorno virtual de Python
- Conexión a Internet la primera vez, para descargar los modelos
- GPU NVIDIA recomendada (al menos 8 GB de VRAM); también funciona en CPU,
  aunque la generación será mucho más lenta

Instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

## Uso

Detecta y elimina automáticamente a la persona más a la izquierda:

```bash
python inpaint.py foto.jpg --output resultado.png
```

Se generan `resultado.png` y `resultado.mask.png`. La máscara es una imagen
en escala de grises: blanco indica la zona que se reconstruirá y negro la zona
que se conservará. Revísala antes de usar el resultado. Para corregirla,
edítala en cualquier editor de imágenes y vuelve a ejecutar:

```bash
python inpaint.py foto.jpg --mask resultado.mask.png --output resultado-final.png
```

El prompt se puede adaptar a la escena; también es posible seleccionar otro
modelo compatible con `StableDiffusionXLInpaintPipeline`:

```bash
python inpaint.py foto.jpg --output resultado.png \
  --prompt "Photorealistic terrace background, matching the original lighting"
```

La detección automática utiliza Mask R-CNN preentrenado en COCO y selecciona
la persona detectada más a la izquierda. La síntesis utiliza el modelo
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. La imagen se procesa con
un lado máximo de 1024 píxeles para reducir el uso de memoria; `--max-size`
permite cambiar ese límite. Las áreas sin máscara se copian de la fotografía
original, sin alteraciones.
