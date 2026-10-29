"""Exercise installed CUDA extension kernels, without loading any checkpoint."""
from datetime import datetime
import json
from pathlib import Path
import torch
import flash_attn
from flash_attn import flash_attn_func
import bitsandbytes as bnb

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    common = {'timestamp': datetime.now().astimezone().isoformat(),
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0)}
    q = torch.randn(1,32,4,64,device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        output = flash_attn_func(q,q,q,causal=False)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(output).all())
    flash = {**common,'version':flash_attn.__version__,'dtype':str(output.dtype),
             'shape':list(output.shape),'finite':finite,'GPU_kernel':'PASS' if finite else 'FAIL'}
    (ROOT/'manifests/flash-attn-kernel-test.json').write_text(json.dumps(flash,indent=2)+'\n')
    layer = bnb.nn.Linear4bit(64,32,bias=False,compute_dtype=torch.bfloat16,quant_type='fp4').cuda()
    with torch.inference_mode():
        output = layer(torch.randn(1,8,64,device='cuda',dtype=torch.bfloat16))
    torch.cuda.synchronize()
    finite_bnb = bool(torch.isfinite(output).all())
    quant = {**common,'bitsandbytes':bnb.__version__,'dtype':str(output.dtype),
             'shape':list(output.shape),'finite':finite_bnb,'status':'PASS' if finite_bnb else 'FAIL'}
    (ROOT/'manifests/bitsandbytes-kernel-test.json').write_text(json.dumps(quant,indent=2)+'\n')
    print(json.dumps({'flash_attention':flash,'bitsandbytes':quant},indent=2))
    if not finite or not finite_bnb:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
