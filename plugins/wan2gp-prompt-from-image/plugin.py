import os
import json

import gradio as gr
from PIL import Image

from shared.utils.plugins import WAN2GPPlugin
from shared.gradio.gallery import get_gradio_file_path


def _extract_prompt_from_image(image_path):
    """Extract a text prompt from image metadata.

    Supports two formats:
    - Forge / SD WebUI: PNG 'parameters' text chunk (positive prompt before
      '\\nNegative prompt:')
    - Wan2GP: JSON in PNG 'comment' text chunk with a 'prompt' key
    """
    if not image_path or not os.path.isfile(image_path):
        return ""
    try:
        with Image.open(image_path) as im:
            # Forge / SD WebUI format: 'parameters' text chunk
            params = (getattr(im, "text", {}) or {}).get("parameters") or im.info.get("parameters")
            if params and isinstance(params, str):
                # Positive prompt is everything before '\nNegative prompt:'
                neg_idx = params.find("\nNegative prompt:")
                if neg_idx > 0:
                    return params[:neg_idx].strip()
                # No negative prompt marker — take up to first '\nSteps:'
                steps_idx = params.find("\nSteps:")
                if steps_idx > 0:
                    return params[:steps_idx].strip()
                return params.strip()

            # Wan2GP format: JSON in 'comment' text chunk
            comment = (getattr(im, "text", {}) or {}).get("comment") or im.info.get("comment")
            if comment and isinstance(comment, str):
                try:
                    data = json.loads(comment)
                    if isinstance(data, dict) and "prompt" in data:
                        return str(data["prompt"]).strip()
                except (json.JSONDecodeError, ValueError):
                    pass

    except Exception:
        pass
    return ""


def _get_first_image_path(gallery_value):
    """Get the file path of the first image in a gallery value."""
    if not gallery_value:
        return ""
    items = gallery_value if isinstance(gallery_value, list) else [gallery_value]
    if not items:
        return ""
    return get_gradio_file_path(items[0]) or ""


class PromptFromImagePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Prompt From Image"
        self.version = "1.0.0"
        self.description = "Auto-load prompts from image metadata when adding start images"

    def setup_ui(self):
        self.request_component("image_start")
        self.request_component("image_start_row")
        self.request_component("prompt")

    def post_ui_setup(self, components):
        image_start = components.get("image_start")
        prompt_component = components.get("prompt")
        if not image_start or not prompt_component:
            return {}

        # Capture references for the closure
        _image_start = image_start
        _prompt = prompt_component

        def create_ui_and_wire():
            """Constructor called by insert_after — runs inside the Gradio
            Blocks context so we can create components and wire events."""
            checkbox = gr.Checkbox(
                label="Auto-load prompt from image metadata",
                value=False,
                info="Reads prompt from Forge/SD WebUI or Wan2GP image metadata",
            )

            def on_image_change(gallery_value, auto_load):
                if not auto_load or not gallery_value:
                    return gr.update()
                path = _get_first_image_path(gallery_value)
                if not path:
                    return gr.update()
                extracted = _extract_prompt_from_image(path)
                if not extracted:
                    return gr.update()
                gr.Info(f"Loaded prompt from image ({len(extracted.split())} words)")
                return gr.update(value=extracted)

            _image_start.change(
                fn=on_image_change,
                inputs=[_image_start, checkbox],
                outputs=[_prompt],
                show_progress=False,
            )

        self.insert_after("image_start_row", create_ui_and_wire)
        return {}
