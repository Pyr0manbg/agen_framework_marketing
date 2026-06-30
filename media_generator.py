import os
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

from google import genai
from google.genai import types
from google.genai.errors import APIError

from huggingface_hub import InferenceClient

from config import (
    GOOGLE_API_KEY,
    GEMINI_IMAGE_MODEL,
    HF_TOKEN,
    OPENAI_API_KEY,
    POLLINATIONS_BASE_URL,
)

# Lazy-loaded SD pipeline (loaded once, reused globally)
_sd_pipeline = None

MEDIA_DIR = Path(__file__).resolve().parent / "generated_media"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

IMAGE_PROMPT_BRAND_CONTEXT = """
Flow Intelligence is a Bulgarian custom software and AI automation agency. Visual brand identity:

COLOR PALETTE (strictly enforce in every prompt):
- Dominant dark background: deep navy/charcoal (like #0f172a or #1e293b - dark slate)
- Accent 1 (primary): teal/turquoise (#0d9488, #14b8a6 - brand main color)
- Accent 2: warm amber/yellow (#f59e0b, #fbbf24 - sparing contrast accent)
- NEVER use bright/sour colors outside this palette — the image must visually belong to the dark teal/yellow brand site

STYLE:
- Modern, tech-forward yet warm and approachable — NOT cold corporate
- Photorealistic photography (NOT illustration, NOT 3D render, NOT flat design)
- Scenes must show a Bulgarian small business context (offices, people working at computers, documents, phones) — NOT generic Silicon Valley tech aesthetic
- Natural lighting combined with warm teal/amber accent lights (e.g. monitor glow, office accent lighting)
- People should look naturally busy, NOT posing or fake stock-photo smiles

FORBIDDEN:
- Futuristic sci-fi aesthetic, neon sour colors, holograms
- Overly abstract/generic "AI technology" visuals (neural networks, circuit diagrams, robots)
- Colors outside the defined palette (no purple, red, bright green as dominant color)
"""


def _ensure_media_dir() -> Path:
    MEDIA_DIR.mkdir(exist_ok=True)
    return MEDIA_DIR


def _generate_filename(prompt: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:40]
    return f"generated_{ts}_{safe.strip('_')}.png"


def _build_pollinations_url(prompt: str, width: int, height: int) -> str:
    encoded = requests.utils.quote(prompt)
    return f"{POLLINATIONS_BASE_URL}/{encoded}?width={width}&height={height}&nologo=true"


def add_logo_watermark(
    image_path: str,
    logo_path: str = "assets/logo.png",
    position: str = "bottom-right",
    logo_width_ratio: float = 0.15,
    margin: int = 20,
    opacity: float = 0.85,
) -> str:
    """Overlay a logo watermark onto an image."""
    logo_path_resolved = Path(logo_path)
    if not logo_path_resolved.is_absolute():
        logo_path_resolved = (Path(__file__).resolve().parent / logo_path).resolve()
    if not logo_path_resolved.exists():
        print(f"[add_logo_watermark] Logo not found at {logo_path_resolved}, skipping")
        return image_path

    base = Image.open(image_path).convert("RGBA")
    logo = Image.open(str(logo_path_resolved)).convert("RGBA")

    base_w, base_h = base.size
    logo_w = int(base_w * logo_width_ratio)
    logo_h = int(logo_w * logo.size[1] / logo.size[0])
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    r, g, b, a = logo.split()
    a = a.point(lambda x: int(x * opacity))
    logo = Image.merge("RGBA", (r, g, b, a))

    pos_map = {
        "bottom-right": (base_w - logo_w - margin, base_h - logo_h - margin),
        "bottom-left": (margin, base_h - logo_h - margin),
        "top-right": (base_w - logo_w - margin, margin),
        "top-left": (margin, margin),
    }
    xy = pos_map.get(position, pos_map["bottom-right"])

    base.paste(logo, xy, logo)
    base = base.convert("RGB")

    base.save(image_path, "PNG")
    return image_path


def _save_image_bytes(image_bytes: bytes, prompt: str) -> str:
    """Save raw image bytes to generated_media/ and apply watermark."""
    out_dir = _ensure_media_dir()
    filename = _generate_filename(prompt)
    out_path = out_dir / filename
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    result = add_logo_watermark(str(out_path.resolve()))
    return result


def _save_pil_image(pil_image: Image.Image, prompt: str) -> str:
    """Save a PIL Image to generated_media/ and apply watermark."""
    out_dir = _ensure_media_dir()
    filename = _generate_filename(prompt)
    out_path = out_dir / filename
    pil_image.save(out_path, "PNG")
    result = add_logo_watermark(str(out_path.resolve()))
    return result


def _try_gemini(prompt: str) -> str | None:
    """Try generating with Gemini Nano Banana. Returns path or None on quota error."""
    if not GOOGLE_API_KEY:
        print("[gemini] GOOGLE_API_KEY not set, skipping")
        return None

    print("[gemini] Calling gemini-2.5-flash-image...")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    try:
        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
    except APIError as e:
        code = str(e)
        if "API_KEY_INVALID" in code or "unauthorized" in code.lower():
            print(f"[gemini] Invalid key: {e}")
            return None
        if "429" in code or "rate" in code.lower() or "quota" in code.lower():
            print("[gemini] Quota exhausted, falling back")
            return None
        if "SAFETY" in code.upper() or "blocked" in code.lower():
            print(f"[gemini] Content policy rejected: {e}")
            return None
        print(f"[gemini] API error (not quota): {e}")
        return None
    except Exception as e:
        print(f"[gemini] Unexpected error: {e}")
        return None

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            print("[gemini] ✅ Image generated")
            return _save_image_bytes(
                part.inline_data.data, prompt
            )

    print("[gemini] Response had no image data")
    return None


def _try_huggingface(prompt: str) -> str | None:
    """Try generating with Hugging Face FLUX.1-dev. Returns path or None."""
    hf_key = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not hf_key:
        print("[huggingface] HF_TOKEN not set, skipping")
        return None

    print("[huggingface] Calling black-forest-labs/FLUX.1-dev via wavespeed...")
    client = InferenceClient(provider="wavespeed", api_key=hf_key)

    for attempt in range(2):
        try:
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev",
            )
        except Exception as e:
            err = str(e).lower()
            if "token" in err or "unauthorized" in err or "invalid" in err:
                print(f"[huggingface] Invalid HF_TOKEN: {e}")
                return None
            if "rate" in err or "429" in err or "too many" in err:
                print(f"[huggingface] Rate limited (attempt {attempt+1}/2)")
                time.sleep(5)
                continue
            print(f"[huggingface] Error (attempt {attempt+1}/2): {e}")
            if attempt == 0:
                time.sleep(3)
                continue
            return None
        break
    else:
        print("[huggingface] Failed after 2 attempts")
        return None

    print("[huggingface] ✅ Image generated")
    return _save_pil_image(image, prompt)


def _try_pollinations(prompt: str, width: int, height: int) -> str | None:
    """Fallback to Pollinations.ai. Returns path or None."""
    print("[pollinations] Calling Pollinations...")
    url = _build_pollinations_url(prompt, width, height)

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
        except requests.exceptions.Timeout:
            print(f"[pollinations] Timeout (attempt {attempt+1}/3)")
            time.sleep(2)
            continue
        except requests.exceptions.ConnectionError:
            print(f"[pollinations] Network error (attempt {attempt+1}/3)")
            time.sleep(2)
            continue

        if response.status_code == 200:
            print("[pollinations] ✅ Image generated")
            return _save_image_bytes(response.content, prompt)

        print(f"[pollinations] HTTP {response.status_code} (attempt {attempt+1}/3)")
        time.sleep(2)

    return None


# ── OpenAI (DALL-E 3 / gpt-image-1) ──────────────────────────


def generate_image_openai(prompt: str, save_local: bool = True) -> str:
    """Generate an image via OpenAI gpt-image-1.

    Uses the REST API directly (not the SDK) for compatibility with the
    gpt-image-1 model which uses b64_json response format.

    Args:
        prompt: Text description of the desired image.
        save_local: If True, saves the image to generated_media/.

    Returns:
        Local file path of the generated (and watermarked) image.

    Raises:
        RuntimeError: If the API call fails or no key is configured.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = "gpt-image-1"
    size = "1024x1024"

    print(f"[openai] Calling {model} (size={size})...")
    t0 = time.time()

    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": model, "prompt": prompt, "n": 1, "size": size},
            timeout=120,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("OpenAI request timed out (120s)")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"OpenAI connection error: {e}")

    if resp.status_code == 400:
        err_data = resp.json()
        msg = (err_data.get("error") or {}).get("message", "")
        m = msg.lower()
        if "content_policy" in m or "safety" in m or "content_filter" in m:
            raise RuntimeError(f"OpenAI content policy violation: {msg}")
        raise RuntimeError(f"OpenAI API error (400): {msg}")
    if resp.status_code == 401:
        raise RuntimeError("OpenAI invalid API key (401)")
    if resp.status_code == 429:
        raise RuntimeError("OpenAI rate limited (429)")
    if resp.status_code == 402:
        raise RuntimeError("OpenAI insufficient credits (402)")
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OpenAI HTTP error: {e}")

    elapsed = time.time() - t0
    body = resp.json()
    image_data = body.get("data", [{}])[0]
    b64_json = image_data.get("b64_json")

    if not b64_json:
        raise RuntimeError("OpenAI returned no b64_json in response")

    import base64
    image_bytes = base64.b64decode(b64_json)

    print(f"[openai] ✅ Image generated in {elapsed:.1f}s")

    result = _save_image_bytes(image_bytes, prompt)
    return result


# ── Local Stable Diffusion 1.5 ───────────────────────────────


def generate_image_local_sd(prompt: str, save_local: bool = True) -> str:
    """Generate an image using local Stable Diffusion v1.5 via diffusers.

    Uses torch.float16, model CPU offload, and attention slicing to fit
    within 4 GB VRAM (RTX 3050). The model is loaded lazily on first call
    and cached globally for subsequent calls.

    Args:
        prompt: Text description of the desired image.
        save_local: If True, saves the image to generated_media/.

    Returns:
        Local file path of the generated (and watermarked) image.

    Raises:
        RuntimeError: If generation fails or CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local SD generation")

    global _sd_pipeline

    if _sd_pipeline is None:
        from diffusers import StableDiffusionPipeline

        print("[local-sd] Loading Stable Diffusion v1.5 pipeline (first call, downloading if needed)...")
        t0 = time.time()

        model_id = "runwayml/stable-diffusion-v1-5"
        hf_key = HF_TOKEN or os.environ.get("HF_TOKEN", "")

        try:
            _sd_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                token=hf_key or None,
            )
        except OSError as e:
            err = str(e).lower()
            if "gated" in err or "access" in err:
                print(f"[local-sd] {model_id} is gated, trying dreamshaper-8...")
                _sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    "Lykon/dreamshaper-8",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )
            else:
                print(f"[local-sd] Failed to load {model_id}: {e}")
                print(f"[local-sd] Trying dreamshaper-8 as fallback...")
                _sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    "Lykon/dreamshaper-8",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )

        _sd_pipeline.enable_model_cpu_offload()
        _sd_pipeline.enable_attention_slicing()
        print(f"[local-sd] Pipeline loaded in {time.time() - t0:.1f}s")

    output_dir = _ensure_media_dir()
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"local_sd_{ts}_{safe.strip('_')}.png"
    out_path = str((output_dir / filename).resolve())

    print(f"[local-sd] Generating image (25 steps)...")
    t0 = time.time()

    try:
        image = _sd_pipeline(
            prompt,
            num_inference_steps=25,
            height=512,
            width=512,
        ).images[0]
    except torch.cuda.OutOfMemoryError:
        print("[local-sd] CUDA out of memory — clearing cache and retrying with 20 steps...")
        torch.cuda.empty_cache()
        try:
            image = _sd_pipeline(
                prompt,
                num_inference_steps=20,
                height=512,
                width=512,
            ).images[0]
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"CUDA OOM even after retry: {e}")

    elapsed = time.time() - t0
    print(f"[local-sd] Image generated in {elapsed:.1f}s")

    if save_local:
        image.save(out_path, "PNG")
        add_logo_watermark(out_path)
        print(f"[local-sd] Saved to {out_path}")
    return out_path


# ── Main generation with fallback ────────────────────────────


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    save_local: bool = True,
) -> str:
    """Generate an image with five-tier fallback: OpenAI → Gemini → HF → Pollinations → local SD.

    Args:
        prompt: Text description of the desired image.
        width: Image width in pixels (used only by Pollinations).
        height: Image height in pixels (used only by Pollinations).
        save_local: Must be True for this implementation.

    Returns:
        Local file path of the generated (and watermarked) image.

    Raises:
        ValueError: If save_local=False.
        RuntimeError: If all providers fail.
    """
    if not save_local:
        raise ValueError("Image generation requires save_local=True")

    # Tier 1: OpenAI DALL-E 3
    if OPENAI_API_KEY:
        print("[generate_image] Trying OpenAI DALL-E 3...")
        try:
            result = generate_image_openai(prompt, save_local)
            if result:
                return result
        except RuntimeError as e:
            print(f"[generate_image] OpenAI failed: {e}")
    else:
        print("[generate_image] OPENAI_API_KEY not set, skipping OpenAI")

    # Tier 2: Gemini Nano Banana
    result = _try_gemini(prompt)
    if result:
        return result

    # Tier 3: Hugging Face FLUX.1-dev
    result = _try_huggingface(prompt)
    if result:
        return result

    # Tier 4: Pollinations
    result = _try_pollinations(prompt, width, height)
    if result:
        return result

    # Tier 5: Local Stable Diffusion 1.5 (no quotas, always works)
    import torch
    if torch.cuda.is_available():
        print("[generate_image] Falling back to local SD 1.5...")
        try:
            result = generate_image_local_sd(prompt, save_local)
            if result:
                return result
        except Exception as e:
            print(f"[generate_image] Local SD also failed: {e}")
    else:
        print("[generate_image] CUDA not available, skipping local SD fallback")

    raise RuntimeError(
        "All image providers failed (OpenAI → Gemini → HuggingFace → Pollinations → local SD)"
    )


def _detect_business_context(text: str) -> str:
    """Detect business type from copy text for scene selection."""
    t = text.lower()
    if any(w in t for w in ["счетовод", "facturi", "книги", "сметк", "данъц", "финанс", "budget", "invoice"]):
        return "accounting"
    if any(w in t for w in ["брокер", "agent", "недвижим", "property", "imot", "имот", "жилищ"]):
        return "real_estate"
    if any(w in t for w in ["retail", "магазин", "shop", "online", "поръчк", "склад", "stock"]):
        return "retail"
    if any(w in t for w in ["услуг", "service", "клиент", "consult", "консулт", "обслужв"]):
        return "services"
    return "general"


def _scene_for_business(business_type: str, platform: str) -> str:
    scenes = {
        "accounting": (
            "a modern accounting office in Bulgaria, two professionals sitting at a desk "
            "with dual monitors showing spreadsheet data and document management software, "
            "morning sunlight streaming through window, coffee cups on desk, "
            "organized paperwork in folders, realistic Bulgarian office interior"
        ),
        "real_estate": (
            "a real estate agency office in Bulgaria, agent showing property photos on a tablet to clients, "
            "modern furnished meeting area, large window with city view, "
            "whiteboard with property listings, warm welcoming atmosphere"
        ),
        "retail": (
            "a small retail shop in Bulgaria, shopkeeper using a tablet or laptop for inventory management, "
            "products neatly arranged on shelves, natural light through shop window, "
            "authentic local store atmosphere, organized workspace"
        ),
        "services": (
            "a service company office in Bulgaria, team of 2-3 people in a meeting around a laptop, "
            "modern open space with plants, natural lighting, "
            "whiteboard with notes in the background, collaborative atmosphere"
        ),
        "general": (
            "a modern small business office in Bulgaria, 30-something professional working at a desk "
            "with a laptop and smartphone, organized workspace with notebook and pen, "
            "morning light, authentic Bulgarian office setting"
        ),
    }
    scene = scenes.get(business_type, scenes["general"])
    if platform.lower() in ("instagram", "tiktok"):
        return scene + ", candid lifestyle photography"
    return scene


def generate_image_prompt_from_copy(final_copy: str, platform: str) -> str:
    """Build a realistic image prompt from post copy and platform.

    Uses template logic (no AI call) — detects business context and
    applies photography-centric terms + Flow Intelligence brand colors.

    Args:
        final_copy: The post copy text.
        platform: Target social platform (linkedin, instagram, etc.).

    Returns:
        An English image prompt for realistic photography-style generation.
    """
    business_type = _detect_business_context(final_copy)
    scene = _scene_for_business(business_type, platform)

    photography = (
        "shot on Canon EOS R5, 35mm lens, natural window lighting, "
        "photojournalistic style, shallow depth of field, candid moment, "
        "realistic skin texture, documentary photography, "
        "color palette dominated by teal (#0d9488) and warm amber (#f59e0b) accents, "
        "modern Bulgarian small business office, authentic real people, no text overlay"
    )

    return f"{IMAGE_PROMPT_BRAND_CONTEXT.strip()}\n{scene}, {photography}"


def test_media_generation() -> None:
    """Test the media generator with three-tier fallback."""
    print("=" * 60)
    print("test_media_generation() — Gemini → HF → Pollinations")
    print("=" * 60)

    scene = (
        "a modern accounting office in Bulgaria, two professionals sitting at a desk "
        "with dual monitors, morning sunlight streaming through window, "
        "coffee cups on desk, organized paperwork in folders"
    )
    photography = (
        "shot on Canon EOS R5, 35mm lens, natural window lighting, "
        "photojournalistic style, shallow depth of field, realistic skin texture, "
        "color palette dominated by teal (#0d9488) and warm amber (#f59e0b) accents, "
        "Bulgarian office interior, authentic real people"
    )
    test_prompt = f"{scene}, {photography}"

    print(f"\n[generate_image] Prompt: {test_prompt[:120]}...")
    print(f"[generate_image] save_local=True")

    try:
        result_path = generate_image(
            prompt=test_prompt,
            width=1024,
            height=1024,
            save_local=True,
        )
    except (PermissionError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        return

    path_obj = Path(result_path)
    if path_obj.exists():
        size_kb = path_obj.stat().st_size / 1024
        print(f"\n  ✓ File saved: {result_path}")
        print(f"  ✓ File size: {size_kb:.1f} KB")
        if path_obj.stat().st_size > 0:
            print(f"  ✓ Image content is non-empty")
        else:
            print(f"  ✗ File is empty (0 bytes)")
    else:
        print(f"\n  ✗ File not found at: {result_path}")

    print(f"\n[generate_image_prompt_from_copy] Testing...")
    test_copy = (
        "AI automation трансформира малкия бизнес. "
        "Открийте как chatbots и workflow automation "
        "спестяват време и пари."
    )
    generated_prompt = generate_image_prompt_from_copy(test_copy, "instagram")
    print(f"  ✓ Generated prompt ({len(generated_prompt)} chars): {generated_prompt}")

    print(f"\n[add_logo_watermark] Testing standalone...")
    if path_obj.exists():
        branded = add_logo_watermark(
            str(path_obj.resolve()),
            logo_path="assets/logo.png",
            position="bottom-right",
            logo_width_ratio=0.15,
            margin=20,
            opacity=0.85,
        )
        print(f"  ✓ Watermark applied: {branded}")

    print("\n" + "=" * 60)
    print("test_media_generation() COMPLETE")
    print("=" * 60)


def test_local_sd() -> None:
    """Test ONLY the local SD generator directly, not through fallback chain."""
    print("=" * 60)
    print("test_local_sd() — Local Stable Diffusion 1.5")
    print("=" * 60)

    prompt = (
        "modern Bulgarian accounting office, two professionals at desks "
        "with monitors, teal and amber color accents, "
        "shot on Canon EOS R5, 35mm lens, natural window lighting, photorealistic"
    )

    print(f"\nPrompt: {prompt}")
    print("Loading SD 1.5 and generating (first run downloads ~4 GB)...\n")

    t_start = time.time()
    try:
        result_path = generate_image_local_sd(prompt=prompt, save_local=True)
    except (RuntimeError, ImportError) as e:
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        return

    total = time.time() - t_start
    path_obj = Path(result_path)
    if path_obj.exists():
        size_kb = path_obj.stat().st_size / 1024
        print(f"\n  ✓ File saved: {result_path}")
        print(f"  ✓ File size: {size_kb:.1f} KB")
        print(f"  ✓ Total time: {total:.1f}s")
        print(f"  ✓ Generation time: {total:.1f}s (includes model loading on first run)")
    else:
        print(f"\n  ✗ File not found at: {result_path}")

    print("\n" + "=" * 60)
    print("test_local_sd() COMPLETE")
    print("=" * 60)


def test_openai_image() -> None:
    """Test ONLY the OpenAI image generator directly, not through fallback chain."""
    print("=" * 60)
    print("test_openai_image() — OpenAI gpt-image-1")
    print("=" * 60)

    prompt = (
        "modern Bulgarian accounting office, two professionals at desks "
        "with monitors, teal and amber color accents, "
        "shot on Canon EOS R5, 35mm lens, natural window lighting, photorealistic"
    )

    print(f"\nPrompt: {prompt}")
    if not OPENAI_API_KEY:
        print("\n  ✗ OPENAI_API_KEY not set in .env")
        print("  Add it and try again.")
        return

    t_start = time.time()
    try:
        result_path = generate_image_openai(prompt=prompt, save_local=True)
    except RuntimeError as e:
        print(f"\n  ✗ Failed: {e}")
        return

    total = time.time() - t_start
    path_obj = Path(result_path)
    if path_obj.exists():
        size_kb = path_obj.stat().st_size / 1024
        print(f"\n  ✓ File saved: {result_path}")
        print(f"  ✓ File size: {size_kb:.1f} KB")
        print(f"  ✓ Total time: {total:.1f}s")
    else:
        print(f"\n  ✗ File not found at: {result_path}")

    print("\n" + "=" * 60)
    print("test_openai_image() COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_media_generation()
