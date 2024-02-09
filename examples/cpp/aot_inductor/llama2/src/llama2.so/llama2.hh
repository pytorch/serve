#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <torch/csrc/inductor/aoti_model_container_runner.h>
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    float *logits; // output logits
    int64_t* toks; // tokens seen so far; no kv-cache :(
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    RunState state; // buffers for the "wave" of activations in the forward pass
    torch::inductor::AOTIModelContainerRunnerCpu *runner;
} Transformer;
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;
void build_transformer(Transformer *t, char* checkpoint_path, int vocab_size, int seq_len);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
float* forward(Transformer* transformer, int token, int pos);
int sample(Sampler* sampler, float* logits);
long time_in_ms();
char* decode(Tokenizer* t, int prev_token, int token);
void free_sampler(Sampler* sampler);
void free_tokenizer(Tokenizer* t);
void free_transformer(Transformer* t);
