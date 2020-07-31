import nvgpu

gpu_info = nvgpu.gpu_info()

model_loaded = False

for info in gpu_info:
    if info['memoryUsed'] > 0:
        model_loaded = True
        break

if not model_loaded:
    exit(1)
