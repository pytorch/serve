import tempfile

import torch.fx._pytree as fx_pytree
from torch._inductor.utils import aot_inductor_launcher, cache_dir
from torch.utils import _pytree as pytree
from torch.utils.cpp_extension import load_inline


def load_exported_model(model_so_path, device):
    module = load_inline(
        name="aot_inductor",
        cpp_sources=[aot_inductor_launcher(model_so_path, device)],
        # use a unique build directory to avoid test interference
        build_directory=tempfile.mkdtemp(dir=cache_dir()),
        functions=["run", "get_call_spec"],
        with_cuda=("cuda" == device),
    )
    call_spec = module.get_call_spec()
    in_spec = pytree.treespec_loads(call_spec[0])
    out_spec = pytree.treespec_loads(call_spec[1])

    def optimized(*args):
        flat_inputs = fx_pytree.tree_flatten_spec((args, {}), in_spec)
        flat_outputs = module.run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized
