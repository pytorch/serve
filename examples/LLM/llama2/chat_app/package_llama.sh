
# Check if the argument is empty or unset
if [ -z "$1" ]; then
  echo "Missing Mandatory argument: Path to llama weights"
  echo "Usage: ./package_llama.sh ./model/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235"
  exit 1
fi

MODEL_GENERATION="true"
LLAMA2_WEIGHTS="$1"

if [ -n "$2" ]; then
  MODEL_GENERATION="$2"
fi

CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

if [ "$MODEL_GENERATION" = "true" ]; then
  echo "Cleaning up previous build of llama-cpp"
  rm -rf build
  git clone https://github.com/ggerganov/llama.cpp.git build
  cd build
  make 
  python -m pip install -r requirements.txt
  
  echo "Convert the 7B model to ggml FP16 format"
  python convert.py $LLAMA2_WEIGHTS --outfile ggml-model-f16.gguf
  
  echo "Quantize the model to 4-bits (using q4_0 method)"
  ./quantize ggml-model-f16.gguf ../ggml-model-q4_0.gguf q4_0
  
  cd ..
  export LLAMA2_Q4_MODEL=$PWD/ggml-model-q4_0.gguf
  echo "Saved quantized model weights to $LLAMA2_Q4_MODEL"
fi

echo "Creating torchserve model archive"
torch-model-archiver --model-name llamacpp --version 1.0 --handler llama_cpp_handler.py --config-file model-config.yaml --archive-format tgz

mkdir -p model_store
mv llamacpp.tar.gz model_store/.
if [ "$MODEL_GENERATION" = "true" ]; then
  echo "Cleaning up build of llama-cpp"
  rm -rf build
fi

