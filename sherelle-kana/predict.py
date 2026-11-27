"""Submission template. Copy this file into your submission folder and fill it in.

The evaluator imports this file and calls, in order:
    model = load_model()                 # once, not timed (limit: 10 minutes)
    text  = predict(model, page_image)   # once per page, timed (limit: 120 s/page)

Rules:
  - `page_image` is ONE page as an RGB PIL.Image rendered at 200 DPI.
  - Return plain text for that page, lines separated by "\n".
  - Weights may be downloaded from Hugging Face inside load_model().
    The scoring run itself happens OFFLINE, after a download pass.
  - No paid APIs, no hard-coded text from the sample page.
"""

import torch

from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

_processors = {}

def load_model() -> object:
    """Load every model your pipeline needs and return them in one object.

    Example (replace with your own pipeline):

        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained("<model-id>")
        model = AutoModelForVision2Seq.from_pretrained("<model-id>").to("cuda").eval()
        return {"processor": processor, "model": model}
    """
    model_A = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", 
        dtype=torch.float16, 
        device_map="cuda:0",
        attn_implementation="sdpa"
                                    )
    return model_A 


def get_processor(max_pixels):
    """Renvoie le processeur Qwen pour une résolution donnée (None = pleine résolution)."""
    if max_pixels not in _processors:
        if max_pixels is None:
            _processors[max_pixels] = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        else:
            _processors[max_pixels] = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-2B-Instruct", max_pixels=max_pixels)
    return _processors[max_pixels]


def predict(model: object, image: Image.Image) -> str:
    """Return the full text of one page image."""
    model = load_model()
    PROMPT_P1 = ("Give an exact transcription in plain text, following the original reading order, "
             "with no comments or formatting, and without correcting any errors.")

    max_pixels=2_500_000
    prompt=PROMPT_P1
    max_new_tokens=1500

    processor = get_processor(max_pixels)
    
    messages = [
        {"role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    model = load_model()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    #t0 = time.perf_counter()

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    torch.cuda.synchronize()
    #seconds = time.perf_counter() - t0
    #vram_gb = torch.cuda.max_memory_allocated() / 1e9

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text = output_text[0]
    tokens = inputs["input_ids"].shape[1]
    generate_tokens = len(generated_ids_trimmed[0])

    return text


if __name__ == "__main__":
    # Quick self-test: python predict.py path/to/page.png
    import sys

    m = load_model()
    img = Image.open(sys.argv[1]).convert("RGB")
    print(predict(m, img))
