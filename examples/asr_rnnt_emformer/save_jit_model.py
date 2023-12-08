import torch
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH


jit_model = 'decoder_jit.pt'
model = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder()

model.eval()

mj = torch.jit.script(model)

mj.save(jit_model)

