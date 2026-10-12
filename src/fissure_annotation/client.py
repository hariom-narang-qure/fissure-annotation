import base64, io, json, os
from pathlib import Path

from PIL import Image
from portkey_ai import Portkey

MODEL = "@applied-ai-openrouter/z-ai/glm-5.3-flash"
PROMPT = (Path(__file__).parents[2] / "prompts" / "fissure_waypoints.txt").read_text()


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(img: Image.Image | list[Image.Image], prompt: str = PROMPT, model: str = MODEL, max_tokens: int = 32000,
        trace_path: Path | None = None) -> str:
    """Stream the reply so a long reasoning run never hangs on one request, and so the thinking
    survives a cut-off: it is appended to `trace_path` as it arrives (reasoning model: max_tokens
    covers the thinking as well as the JSON)."""
    client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    images = img if isinstance(img, list) else [img]
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": png_data_url(i)}} for i in images]
    stream = client.chat.completions.create(model=model, max_completion_tokens=max_tokens, stream=True,
                                            messages=[{"role": "user", "content": content}])
    answer, finish = [], None
    trace = open(trace_path, "w") if trace_path else None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if trace and getattr(delta, "reasoning", None):
            trace.write(delta.reasoning)
            trace.flush()
        answer.append(delta.content or "")
        finish = chunk.choices[0].finish_reason or finish
    if trace:
        trace.close()
    if not "".join(answer):
        raise RuntimeError(f"no content, finish_reason={finish}: thinking likely used all {max_tokens} tokens")
    return "".join(answer)


def parse_json(reply: str) -> dict:
    "last {...} block, in case the model wraps the JSON in prose or a code fence"
    start, end = reply.find("{"), reply.rfind("}")
    return json.loads(reply[start:end + 1])


def propose_fissures(img: Image.Image, trace_path: Path | None = None) -> dict:
    return parse_json(ask(img, trace_path=trace_path))
