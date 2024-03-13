#!/bin/bash
batch_size=$1

start_time=$(date +%s%N)
for _ in `seq 1 10`; do
for i in `seq 1 $batch_size`; do
  curl http://localhost:8080/predictions/codegen25 -T ./sample_text_0.txt -o output_${i}.txt &
done
wait;
done
end_time=$(date +%s%N)
elapsed=$(($(($end_time - $start_time))/10000000))
echo "Average e2e runtime per batch for batch size = ${batch_size} is ${elapsed} ms" 
