VK IA - Boîte à outils IA pour le traitement d'images et le graphisme.

Installation :
    pip install pillow opencv-python numpy rembg onnxruntime
    # Optionnel (génération d'images par IA, GPU conseillé) :
    pip install torch diffusers transformers accelerate

Utilisation :
    python vk_ia.py --help
    python vk_ia.py auto photo.jpg -o sortie.jpg
    python vk_ia.py remove-bg photo.jpg -o detoure.png
    python vk_ia.py replace-bg photo.jpg --color "#00A859" -o fond_vert.jpg
    python vk_ia.py generate "un lion au coucher du soleil, style affiche" -o lion.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------- Entrées / sorties ----------
def load(path, mode="RGB"):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respecte l'orientation des photos
    return img.convert(mode)


def save(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg") and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, quality=95)
    print(f"✅ Enregistré : {path}")


def cv():
    try:
        import cv2
        return cv2
    except ImportError:
        raise SystemExit("Installe OpenCV : pip install opencv-python")


def to_cv(img):
    return cv().cvtColor(np.array(img.convert("RGB")), cv().COLOR_RGB2BGR)


def from_cv(arr):
    return Image.fromarray(cv().cvtColor(arr, cv().COLOR_BGR2RGB))


# ---------- Outils de base ----------
def resize(img, width=None, height=None):
    w, h = img.size
    if width and not height:
        height = round(h * width / w)
    elif height and not width:
        width = round(w * height / h)
    return img.resize((width or w, height or h), Image.LANCZOS)


def adjust(img, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    for enhancer, value in (
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast, contrast),
        (ImageEnhance.Color, saturation),
        (ImageEnhance.Sharpness, sharpness),
    ):
        img = enhancer(img).enhance(value)
    return img


def auto_enhance(img):
    """Amélioration automatique : contraste, couleurs, netteté."""
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Color(img).enhance(1.12)
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=110, threshold=3))


def denoise(img, strength=7):
    return from_cv(cv().fastNlMeansDenoisingColored(to_cv(img), None, strength, strength, 7, 21))


def upscale(img, factor=2):
    """Agrandissement + netteté. Pour du vrai upscaling IA, voir Real-ESRGAN."""
    big = img.resize((img.width * factor, img.height * factor), Image.LANCZOS)
    return big.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=2))


# ---------- Détourage / fonds (IA) ----------
def remove_bg(img):
    try:
        from rembg import remove
    except ImportError:
        raise SystemExit("Installe rembg : pip install rembg onnxruntime")
    return remove(img.convert("RGBA"))


def replace_bg(img, color=None, bg_path=None):
    cut = remove_bg(img)
    if bg_path:
        bg = load(bg_path).resize(cut.size, Image.LANCZOS).convert("RGBA")
    else:
        bg = Image.new("RGBA", cut.size, color or "#FFFFFF")
    bg.alpha_composite(cut)
    return bg.convert("RGB")


# ---------- Effets artistiques ----------
def black_white(img):
    return ImageOps.grayscale(img).convert("RGB")


def sepia(img):
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    arr = np.clip(np.array(img, dtype=np.float32) @ m.T, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def vignette(img, strength=0.6):
    w, h = img.size
    x = np.linspace(-1, 1, w)[None, :]
    y = np.linspace(-1, 1, h)[:, None]
    mask = 1 - strength * np.clip(np.sqrt(x**2 + y**2) - 0.3, 0, 1)
    arr = np.array(img, dtype=np.float32) * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def sketch(img):
    gray = cv().cvtColor(to_cv(img), cv().COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv().GaussianBlur(inv, (21, 21), 0)
    out = cv().divide(gray, 255 - blur, scale=256)
    return Image.fromarray(out).convert("RGB")


def cartoon(img):
    c = cv()
    bgr = to_cv(img)
    gray = c.medianBlur(c.cvtColor(bgr, c.COLOR_BGR2GRAY), 7)
    edges = c.adaptiveThreshold(gray, 255, c.ADAPTIVE_THRESH_MEAN_C, c.THRESH_BINARY, 9, 9)
    color = c.bilateralFilter(bgr, 9, 250, 250)
    return from_cv(c.bitwise_and(color, color, mask=edges))


# ---------- Vision (détection) ----------
def blur_faces(img, strength=51):
    c = cv()
    bgr = to_cv(img)
    cascade = c.CascadeClassifier(c.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(c.cvtColor(bgr, c.COLOR_BGR2GRAY), 1.1, 5)
    k = strength | 1  # noyau impair
    for (x, y, w, h) in faces:
        bgr[y:y + h, x:x + w] = c.GaussianBlur(bgr[y:y + h, x:x + w], (k, k), 0)
    print(f"👤 {len(faces)} visage(s) flouté(s)")
    return from_cv(bgr)


# ---------- Graphisme ----------
def watermark(img, text, opacity=120):
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = max(base.width // 20, 14)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    pos = (base.width - (box[2] - box[0]) - 20, base.height - (box[3] - box[1]) - 20)
    draw.text(pos, text, font=font, fill=(255, 255, 255, opacity))
    return Image.alpha_composite(base, layer).convert("RGB")


def collage(paths, cols=2, cell=600):
    imgs = [ImageOps.fit(load(p), (cell, cell), Image.LANCZOS) for p in paths]
    rows = -(-len(imgs) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * cell, (i // cols) * cell))
    return sheet


# ---------- Génération d'images (IA) ----------
def generate(prompt, steps=4, size=512):
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        raise SystemExit("Installe : pip install torch diffusers transformers accelerate")
    gpu = torch.cuda.is_available()
    kwargs = {"torch_dtype": torch.float16, "variant": "fp16"} if gpu else {}
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", **kwargs)
    pipe = pipe.to("cuda" if gpu else "cpu")
    return pipe(prompt, num_inference_steps=steps, guidance_scale=0.0,
                width=size, height=size).images[0]


# ---------- Interface en ligne de commande ----------
def build_parser():
    p = argparse.ArgumentParser(prog="vk_ia", description="VK IA - outils IA pour images")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cmd(name, help_, *extra):
        s = sub.add_parser(name, help=help_)
        if name not in ("generate", "collage"):
            s.add_argument("input", help="image source (ou dossier pour 'batch')")
        s.add_argument("-o", "--output", required=True, help="fichier de sortie")
        for args, kw in extra:
            s.add_argument(*args, **kw)
        return s

    cmd("auto", "amélioration automatique")
    cmd("resize", "redimensionner", (["--width"], {"type": int}), (["--height"], {"type": int}))
    cmd("adjust", "réglages manuels",
        (["--brightness"], {"type": float, "default": 1.0}),
        (["--contrast"], {"type": float, "default": 1.0}),
        (["--saturation"], {"type": float, "default": 1.0}),
        (["--sharpness"], {"type": float, "default": 1.0}))
    cmd("denoise", "réduction du bruit", (["--strength"], {"type": int, "default": 7}))
    cmd("upscale", "agrandir", (["--factor"], {"type": int, "default": 2}))
    cmd("remove-bg", "supprimer l'arrière-plan (IA)")
    cmd("replace-bg", "remplacer l'arrière-plan (IA)",
        (["--color"], {"default": "#FFFFFF"}), (["--image"], {"help": "image de fond"}))
    cmd("bw", "noir et blanc")
    cmd("sepia", "effet sépia")
    cmd("vignette", "effet vignette", (["--strength"], {"type": float, "default": 0.6}))
    cmd("sketch", "effet crayon")
    cmd("cartoon", "effet dessin animé")
    cmd("blur-faces", "flouter les visages")
    cmd("watermark", "ajouter un filigrane", (["--text"], {"required": True}))
    cmd("collage", "créer un collage",
        (["images"], {"nargs": "+"}), (["--cols"], {"type": int, "default": 2}))
    cmd("generate", "générer une image depuis un texte (IA)",
        (["prompt"], {}), (["--steps"], {"type": int, "default": 4}))
    cmd("batch", "améliorer automatiquement tout un dossier")
    return p


def main():
    a = build_parser().parse_args()
    c = a.cmd

    if c == "generate":
        return save(generate(a.prompt, a.steps), a.output)
    if c == "collage":
        return save(collage(a.images, a.cols), a.output)
    if c == "batch":
        out = Path(a.output)
        for f in sorted(Path(a.input).iterdir()):
            if f.suffix.lower() in EXTS:
                save(auto_enhance(load(f)), out / f.name)
        return

    img = load(a.input)
    ops = {
        "auto": lambda: auto_enhance(img),
        "resize": lambda: resize(img, a.width, a.height),
        "adjust": lambda: adjust(img, a.brightness, a.contrast, a.saturation, a.sharpness),
        "denoise": lambda: denoise(img, a.strength),
        "upscale": lambda: upscale(img, a.factor),
        "remove-bg": lambda: remove_bg(img),
        "replace-bg": lambda: replace_bg(img, a.color, a.image),
        "bw": lambda: black_white(img),
        "sepia": lambda: sepia(img),
        "vignette": lambda: vignette(img, a.strength),
        "sketch": lambda: sketch(img),
        "cartoon": lambda: cartoon(img),
        "blur-faces": lambda: blur_faces(img),
        "watermark": lambda: watermark(img, a.text),
    }
    save(ops[c](), a.output)


if __name__ == "__main__":
    main()
