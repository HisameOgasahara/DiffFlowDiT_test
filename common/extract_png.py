import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

def read_metadata(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG 파일이 아닙니다.")
    offset, metadata = 8, {}
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        chunk = data[offset + 8:offset + 8 + length]
        offset += length + 12
        if kind == b"tEXt":
            key, value = chunk.split(b"\0", 1)
        elif kind == b"zTXt":
            key, packed = chunk.split(b"\0", 1)
            value = zlib.decompress(packed[1:])
        elif kind == b"iTXt":
            key, rest = chunk.split(b"\0", 1)
            compressed, method = rest[:2]
            language, translated, value = rest[2:].split(b"\0", 2)
            if compressed:
                value = zlib.decompress(value)
        else:
            continue
        if key in {b"prompt", b"workflow"}:
            metadata[key.decode()] = json.loads(value.decode("utf-8"))
    if "prompt" not in metadata:
        raise ValueError("PNG에 ComfyUI prompt 메타데이터가 없습니다.")
    return metadata

def extract_config(metadata):
    nodes = metadata["prompt"]
    samplers = [node for node in nodes.values() if node["class_type"] == "KSampler"]
    if len(samplers) != 1:
        raise ValueError("이 비교 도구는 KSampler 1개인 기본 txt2img 워크플로를 대상으로 합니다.")
    values = samplers[0]["inputs"]
    def linked(key):
        return nodes[str(values[key][0])]
    model = linked("model")
    positive, negative, latent = linked("positive"), linked("negative"), linked("latent_image")
    if model["class_type"] != "UNETLoader" or latent["class_type"] != "EmptyLatentImage":
        raise ValueError("모델 패치 또는 img2img 경로는 별도로 분석해야 합니다.")
    for node in (positive, negative):
        if node["class_type"] != "CLIPTextEncode":
            raise ValueError("기본 CLIPTextEncode 경로만 지원합니다.")
    clip = nodes[str(positive["inputs"]["clip"][0])]["inputs"]
    decoders = [n for n in nodes.values() if n["class_type"] == "VAEDecode"]
    vae = nodes[str(decoders[0]["inputs"]["vae"][0])]["inputs"]
    return {
        "prompt": positive["inputs"]["text"], "negative_prompt": negative["inputs"]["text"],
        **{k: values[k] for k in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise")},
        **{k: latent["inputs"][k] for k in ("width", "height", "batch_size")},
        "models": {"dit": model["inputs"]["unet_name"], "text_encoder": clip["clip_name"], "vae": vae["vae_name"]},
        "clip_type": clip["type"], "lora_applied": False,
    }

def export(path, destination):
    metadata = read_metadata(path)
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in {**metadata, "generation": extract_config(metadata)}.items():
        (destination / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "source_image.json").write_text(json.dumps({"filename": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.image, args.destination)
