"""기존 모델 설정과 원본 DiffSynth 토크나이저 설정으로 파일을 준비한다."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, read_json, write_json, sha256


def prepare(models, config, runtime):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    hp = read_json(config)
    options = read_json(runtime)
    source = read_json(ROOT / "config/model_source.json")
    models.mkdir(parents=True, exist_ok=True)
    manifest_path = models / "model_manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    api = HfApi(token=False)
    revision = api.model_info(source["repository"], revision=source["revision"]).sha
    manifest = {"repository": source["repository"], "revision": revision, "weights": {},
                "tokenizers": {}, "tokenizer_sources": {}}
    for kind, filename in hp["models"].items():
        downloaded = hf_hub_download(source["repository"], source["folders"][kind] + "/" + filename,
                                     revision=revision, token=False)
        target = models / "weights" / Path(filename).name
        target.parent.mkdir(exist_ok=True)
        digest = sha256(downloaded)
        if not target.exists() or sha256(target) != digest:
            shutil.copy2(downloaded, target)
        manifest["weights"][kind] = {"filename": target.name, "sha256": digest, "bytes": target.stat().st_size}
    for name, spec in options["tokenizers"].items():
        if "local_path" in spec:
            folder = ROOT / spec["local_path"]
            source_info = dict(spec, path=str(folder.resolve()))
        else:
            previous = old.get("tokenizer_sources", {}).get(name, {})
            same_spec = all(previous.get(key) == value for key, value in spec.items())
            revision = previous["resolved_revision"] if same_spec else api.model_info(spec["repository"], revision=spec["revision"]).sha
            prefix = spec["subfolder"].strip("/")
            patterns = [prefix + "/*"] if prefix else ["*.json", "*.txt", "*.model", "*.jinja"]
            folder = Path(snapshot_download(spec["repository"], revision=revision, allow_patterns=patterns, token=False)) / prefix
            source_info = dict(spec, resolved_revision=revision, path=str(folder))
        manifest["tokenizers"][name] = {p.relative_to(folder).as_posix(): sha256(p)
                                         for p in folder.rglob("*") if p.is_file()}
        manifest["tokenizer_sources"][name] = source_info
    write_json(manifest_path, manifest)
    print("원본 DiffSynth 모델·토크나이저 준비 완료:", models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/generation.json")
    parser.add_argument("--runtime", type=Path, default=ROOT / "diffsynth/original_runtime.json")
    args = parser.parse_args()
    prepare(args.models, args.config, args.runtime)
