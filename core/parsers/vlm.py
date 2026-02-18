"""
VLM (Visual Language Model) Parser Module

Uses Ollama's vision-capable models (e.g., qwen2.5-vl) to extract
structured text from presentation slides, complex PDF pages, and
other visually-rich documents that standard OCR misses.

Follows the project's async httpx pattern (see core/llm/unload_ollama_model.py).
"""

import httpx
import base64
import time
import traceback

from core.config import settings
from core.constants import PORT1, VLM_MODEL

LOCAL_BASE_URL = settings.LOCAL_BASE_URL

# Prompt tuned for slide/document extraction
VLM_EXTRACTION_PROMPT = (
    "You are an intelligent document parser. "
    "Analyze this image and extract ALL content into structured Markdown.\n"
    "Rules:\n"
    "- Transcribe every text element: titles, headers, bullet points, paragraphs, footnotes.\n"
    "- If there are tables, output them as proper Markdown tables.\n"
    "- If there are charts or diagrams, describe the data points, axes, labels, and trends.\n"
    "- Preserve the logical reading order (top-to-bottom, left-to-right).\n"
    "- Do NOT describe visual styling (colors, fonts, layout) unless it conveys meaning.\n"
    "- Output ONLY the extracted Markdown. No filler text like 'Here is the content'."
)


def _encode_image_base64(image_input) -> str:
    """Encode an image file path or raw bytes to a base64 string."""
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image_input, bytes):
        return base64.b64encode(image_input).decode("utf-8")
    else:
        raise TypeError(f"Expected str (path) or bytes, got {type(image_input)}")


async def vlm_parse_slide(image_input, port: int = PORT1) -> str:
    """
    Send an image to Ollama VLM for structured text extraction.

    Args:
        image_input: File path (str) or raw PNG bytes.
        port: Ollama API port (default: PORT1 from constants).

    Returns:
        Extracted Markdown string, or "" on failure.
    """
    try:
        start_time = time.time()
        image_b64 = _encode_image_base64(image_input)

        url = f"{LOCAL_BASE_URL}:{port}/api/generate"

        payload = {
            "model": VLM_MODEL,
            "prompt": VLM_EXTRACTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,   # Low temp for factual extraction
                "num_ctx": 4096,      # Sufficient for most slides
            },
        }

        print(f"[VLM] Sending page to Ollama ({VLM_MODEL}) on port {port}...")

        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        result = response.json()
        content = result.get("response", "").strip()

        elapsed = time.time() - start_time
        print(f"[VLM] Completed in {elapsed:.2f}s  |  {len(content)} chars extracted.")

        return content

    except httpx.ConnectError:
        print(
            f"[VLM] Connection refused at {LOCAL_BASE_URL}:{port}. "
            "Is Ollama running? Try: ollama serve"
        )
        return ""
    except httpx.TimeoutException:
        print(f"[VLM] Request timed out after 180s for model {VLM_MODEL}.")
        return ""
    except httpx.HTTPStatusError as e:
        print(f"[VLM] HTTP error: {e}")
        return ""
    except Exception as e:
        print(f"[VLM] Unexpected error: {e}")
        traceback.print_exc()
        return ""


# ---------------------------------------------------------------------------
# Batch prompt: instructs the model to output per-page content with delimiters
# ---------------------------------------------------------------------------
VLM_BATCH_PROMPT_TEMPLATE = (
    "You are an intelligent document parser. "
    "You are given {count} page images from a document.\n"
    "Extract ALL content from EACH page into structured Markdown.\n\n"
    "Rules:\n"
    "- Transcribe every text element: titles, headers, bullet points, paragraphs, footnotes.\n"
    "- If there are tables, output them as proper Markdown tables.\n"
    "- If there are charts or diagrams, describe the data points, axes, labels, and trends.\n"
    "- Preserve the logical reading order (top-to-bottom, left-to-right).\n"
    "- Do NOT describe visual styling (colors, fonts, layout) unless it conveys meaning.\n\n"
    "CRITICAL FORMAT RULE:\n"
    "Separate the content of each page using the following delimiter on its own line:\n"
    "---PAGE N---\n"
    "where N is the page number (1, 2, 3, ...).\n\n"
    "Example output:\n"
    "---PAGE 1---\n"
    "# Slide Title\n"
    "- Bullet point content...\n\n"
    "---PAGE 2---\n"
    "## Another Title\n"
    "Paragraph text...\n\n"
    "Begin extraction now. Output ONLY the extracted Markdown with page delimiters."
)


def _split_batch_response(response_text: str, expected_count: int) -> list[str]:
    """
    Parse a batch VLM response into per-page strings using ---PAGE N--- delimiters.

    Returns a list of length `expected_count`. Missing pages get empty strings.
    """
    import re

    pages = [""] * expected_count

    # Split on ---PAGE N--- pattern (case-insensitive, flexible whitespace)
    parts = re.split(r"---\s*PAGE\s+(\d+)\s*---", response_text, flags=re.IGNORECASE)

    # parts alternates: [preamble, page_num, content, page_num, content, ...]
    i = 1  # skip preamble (text before first delimiter)
    while i + 1 < len(parts):
        try:
            page_num = int(parts[i])
            content = parts[i + 1].strip()
            if 1 <= page_num <= expected_count:
                pages[page_num - 1] = content
        except (ValueError, IndexError):
            pass
        i += 2

    return pages


async def vlm_parse_batch(
    images: list[bytes],
    page_labels: list[str] | None = None,
    port: int = PORT1,
    batch_size: int = 10,
) -> list[str]:
    """
    Send multiple page images in batched VLM calls for efficient extraction.

    Args:
        images: List of raw PNG bytes, one per page.
        page_labels: Optional labels for logging (e.g. ["Page 1", "Slide 3"]).
        port: Ollama API port (default: PORT1 from constants).
        batch_size: Max images per VLM API call (default: 10).

    Returns:
        List of extracted Markdown strings, one per input image.
        Empty string for pages where extraction failed.
    """
    if not images:
        return []

    total = len(images)
    results = [""] * total
    labels = page_labels or [f"Page {i+1}" for i in range(total)]

    # Chunk images into batches
    batches = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batches.append((start, end))

    print(f"[VLM] Batch processing {total} pages in {len(batches)} batch(es) (batch_size={batch_size})")

    url = f"{LOCAL_BASE_URL}:{port}/api/generate"

    for batch_idx, (start, end) in enumerate(batches):
        batch_images = images[start:end]
        batch_count = len(batch_images)
        batch_labels = labels[start:end]

        print(f"[VLM] Batch {batch_idx+1}/{len(batches)}: {batch_labels[0]} to {batch_labels[-1]} ({batch_count} pages)")

        try:
            start_time = time.time()

            # Encode all images in this batch
            encoded_images = [_encode_image_base64(img) for img in batch_images]

            prompt = VLM_BATCH_PROMPT_TEMPLATE.format(count=batch_count)

            payload = {
                "model": VLM_MODEL,
                "prompt": prompt,
                "images": encoded_images,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 32768,  # Larger context for multi-page extraction
                },
            }

            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            result = response.json()
            content = result.get("response", "").strip()

            elapsed = time.time() - start_time
            print(f"[VLM] Batch {batch_idx+1} completed in {elapsed:.2f}s  |  {len(content)} chars total")

            # Parse the delimited response into per-page strings
            page_texts = _split_batch_response(content, batch_count)

            # Check parsing quality
            non_empty = sum(1 for t in page_texts if t)
            if non_empty == 0 and content:
                # Delimiter parsing failed completely — assign all content to first page
                # and fall back to per-page processing for the rest
                print(f"[VLM] Warning: Delimiter parsing failed for batch {batch_idx+1}. "
                      f"Falling back to assigning all content to first page.")
                page_texts[0] = content
            else:
                print(f"[VLM] Batch {batch_idx+1}: {non_empty}/{batch_count} pages extracted successfully")

            # Map results back to the global results array
            for i, text in enumerate(page_texts):
                results[start + i] = text

        except httpx.ConnectError:
            print(
                f"[VLM] Connection refused at {LOCAL_BASE_URL}:{port}. "
                "Is Ollama running? Try: ollama serve"
            )
        except httpx.TimeoutException:
            print(f"[VLM] Batch {batch_idx+1} timed out after 300s.")
        except httpx.HTTPStatusError as e:
            print(f"[VLM] HTTP error in batch {batch_idx+1}: {e}")
        except Exception as e:
            print(f"[VLM] Unexpected error in batch {batch_idx+1}: {e}")
            traceback.print_exc()

    # Summary
    total_extracted = sum(1 for r in results if r)
    print(f"[VLM] Batch processing complete: {total_extracted}/{total} pages extracted")
    return results
