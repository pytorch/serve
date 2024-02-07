find_library(SENTENCEPIECE sentencepiece)
add_library(bert_handler SHARED src/bert_handler.cc)
target_link_libraries(bert_handler PRIVATE ts_backends_core ts_utils ${TORCH_LIBRARIES} ${SENTENCEPIECE})
