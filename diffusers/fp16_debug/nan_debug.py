"""모듈 입출력에서 최초 NaN/Inf를 기록하고 중단한다. 텐서를 변경하지 않는다."""
import json
from pathlib import Path
import torch


class NanDebugger:
    def __init__(self, output):
        self.output = Path(output) / "nan_debug"
        self.output.mkdir(parents=True, exist_ok=True)
        self.handles = []
        self.calls = 0
        self.last_finite = None

    def check(self, label, value):
        if isinstance(value, dict):
            for key, item in value.items():
                self.check(f"{label}.{key}", item)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self.check(f"{label}[{i}]", item)
        elif isinstance(value, torch.Tensor) and value.is_floating_point():
            self.calls += 1
            finite = torch.isfinite(value)
            if not finite.all().item():
                report = dict(location=label, shape=list(value.shape), dtype=str(value.dtype),
                              nan_count=value.isnan().sum().item(), inf_count=value.isinf().sum().item(),
                              checked_tensors=self.calls, last_finite=self.last_finite)
                (self.output / "first_nonfinite.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                torch.save(value.detach().cpu(), self.output / "first_nonfinite.pt")
                raise FloatingPointError(f"NaN/Inf: {label}; 기록: {self.output}")
            self.last_finite = label

    def attach(self, name, model):
        for path, module in model.named_modules():
            label = f"{name}.{path}" if path else name
            def before(module, args, kwargs, label=label):
                self.check(label + ".input", (args, kwargs))
            def after(module, args, kwargs, output, label=label):
                self.check(label + ".output", output)
            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(after, with_kwargs=True))
        return self.handles
