#!/bin/bash

for i in {1..64}; do
    python ../test_stream_response.py > t_${i} &
done
