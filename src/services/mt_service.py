"""Machine Translation Service using MADLAD-400-3B on Modal.com."""

import time
from typing import Any

import modal

# Modal app configuration
app = modal.App("unitra-ml")

# Container image with ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
        "accelerate",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
class MTService:
    """Machine translation service using MADLAD-400-3B model."""

    MODEL_ID = "google/madlad400-3b-mt"

    @modal.enter()
    def load_model(self) -> None:
        """Load the model when container starts."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    @modal.method()
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 256,
    ) -> dict[str, Any]:
        """Translate text from source language to target language.

        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)
            max_length: Maximum output length

        Returns:
            Dictionary with translated_text, source_lang, target_lang, latency_ms
        """
        import torch

        start_time = time.perf_counter()

        # MADLAD-400 uses language tags like <2zh> for Chinese
        input_text = f"<2{target_lang}> {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "latency_ms": latency_ms,
        }

    @modal.method()
    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "model": self.MODEL_ID}


# Web endpoint for HTTP access
@app.function(image=image)
@modal.web_endpoint(method="POST")
def translate_endpoint(request: dict[str, Any]) -> dict[str, Any]:
    """HTTP endpoint for translation."""
    service = MTService()
    return service.translate(
        text=request["text"],
        source_lang=request.get("source_lang", "auto"),
        target_lang=request["target_lang"],
    )


if __name__ == "__main__":
    # Local testing
    with app.run():
        service = MTService()
        result = service.translate(
            text="Hello, how are you?",
            source_lang="en",
            target_lang="zh",
        )
        print(result)
