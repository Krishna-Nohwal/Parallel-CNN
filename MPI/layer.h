#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#ifndef LAYER_H
#define LAYER_H

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

typedef struct Layer {
    int M, N, O;
    float *output, *preact, *bias, *weight;
    float *d_output, *d_preact, *d_weight;
} Layer;

static void layer_init(Layer *layer, int M, int N, int O) {
    layer->M = M;
    layer->N = N;
    layer->O = O;
    layer->output = (float *)calloc((size_t)O, sizeof(float));
    layer->preact = (float *)calloc((size_t)O, sizeof(float));
    layer->bias = (float *)calloc((size_t)N, sizeof(float));
    layer->weight = (float *)calloc((size_t)M * (size_t)N, sizeof(float));
    layer->d_output = (float *)calloc((size_t)O, sizeof(float));
    layer->d_preact = (float *)calloc((size_t)O, sizeof(float));
    layer->d_weight = (float *)calloc((size_t)M * (size_t)N, sizeof(float));
    for (int i = 0; i < N; ++i) {
        layer->bias[i] = 0.5f - (float)rand() / RAND_MAX;

        for (int j = 0; j < M; ++j) {
            layer->weight[i * M + j] = 0.5f - (float)rand() / RAND_MAX;
        }
    }
}

static void layer_destroy(Layer *layer) {
    free(layer->output);
    free(layer->preact);
    free(layer->bias);
    free(layer->weight);
    free(layer->d_output);
    free(layer->d_preact);
    free(layer->d_weight);
}

static void layer_set_output(Layer *layer, const float *data) {
    memcpy(layer->output, data, sizeof(float) * (size_t)layer->O);
}

static void layer_clear(Layer *layer) {
    memset(layer->output, 0, sizeof(float) * (size_t)layer->O);
    memset(layer->preact, 0, sizeof(float) * (size_t)layer->O);
}

static void layer_bp_clear(Layer *layer) {
    memset(layer->d_weight, 0, sizeof(float) * (size_t)layer->M * (size_t)layer->N);
}

float step_function(float v) {
    return 1 / (1 + exp(-v));
}

void apply_step_function(float *input, float *output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = step_function(input[i]);
    }
}

void makeError(float *err, float *output, unsigned int Y, int N) {
    for (int i = 0; i < N; ++i) {
        err[i] = (i == Y) ? 1.0f - output[i] : -output[i];
    }
}

void apply_grad(float *output, float *grad, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] += dt * grad[i];
    }
}

void flattenTensor(float* flattenedTensor, float*** tensor, int numOutputs, int outputDim1, int outputDim2) {
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        flattenedTensor[i] = tensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2];
        ++idx;
    }
}

void buildTensor62424(float builtTensor[6][24][24], float flattenedTensor[3456]) {
    int numOutputs = 6;
    int outputDim1 = 24;
    int outputDim2 = 24;
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

void buildTensor666(float builtTensor[6][6][6], float flattenedTensor[216]){ 
    int numOutputs = 6;
    int outputDim1 = 6;
    int outputDim2 = 6; 
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

void buildTensor655(float builtTensor[6][5][5], float flattenedTensor[150]){ 
    int numOutputs = 6;
    int outputDim1 = 5;
    int outputDim2 = 5; 
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

void buildTensor144(float builtTensor[1][4][4], float flattenedTensor[16]){ 
    int numOutputs = 1;
    int outputDim1 = 4;
    int outputDim2 = 4; 
    int idx = 0;
    for (int i = 0; i < numOutputs * outputDim1 * outputDim2; ++i) {
        builtTensor[idx / (outputDim1 * outputDim2)][idx / outputDim2 % outputDim1][idx % outputDim2] = flattenedTensor[i];
        ++idx;
    }
}

int Min(int a, int b){
    if (a>b)
        return b;
    return a;
}

void fp_preact_c1(const float input[28][28], float preact[6][24][24], const float weight[6][5][5]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    float placeHolder[6 * 24 * 24];
    float result[6 * 24 * 24];

    int S = 6 * 24 * 24;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }
    
    for (int l = start; l < end; ++l) {
        int m = l / (24 * 24);
        int x = (l / 24) % 24;
        int y = l % 24;
        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                sum += input[x + i][y + j] * weight[m][i][j];
            }
        }
        placeHolder[l] += sum;
    }
    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor62424(preact, result);
}



void fp_bias_c1(float preact[6][24][24], const float bias[6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 24 * 24;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l] = 0;
    }

    for (int l = start; l < end; ++l) {
        int m = l / (24 * 24);
        int x = (l / 24) % 24;
        int y = l % 24;
        placeHolder[l] = preact[m][x][y] + bias[m];
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor62424(preact, result);
}

void fp_preact_s1(const float input[6][24][24], float preact[6][6][6], const float weight[1][4][4]) {
    
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 6 * 6;
    float placeHolder[S];
    float result[S];

    int it = ceil(6*6*6/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }
    
    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int x = (l / 6) % 6;
        int y = l % 6;
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {  
            for (int j = 0; j < 4; ++j) {  
                sum += weight[0][i][j] * input[m][x * 4 + i][y * 4 + j];
            }
        }
        placeHolder[l] += sum;
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor666(preact, result);
}

void fp_bias_s1(float preact[6][6][6], const float bias[1]) {

    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    float placeHolder[S];
    float result[S];
    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l]=0;
    }

    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int j = (l / 6) % 6;
        int k = l % 6;
        placeHolder[l] = preact[m][j][k] + bias[0];
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor666(preact, result);
}


void fp_preact_f(const float input[6][6][6], float preact[10], const float weight[10][6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 10;
    float placeHolder[S];

    
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
    }

    it = ceil(10 * 6 * 6 * 6/(float)p);
    start = r * it;
    end = Min(start + it, 10 * 6 * 6 * 6);
    
    for (int i = start; i < end; ++i) {
        int m = i / (6 * 6 * 6);
        int j = (i / (6 * 6)) % 6;
        int k = (i / 6) % 6;
        int l = i % 6;
        placeHolder[m] += weight[m][j][k][l] * input[j][k][l];
    }

    MPI_Allreduce(placeHolder, preact, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}


void fp_bias_f(float preact[10], const float bias[10]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);


    int S = 10;
    float placeHolder[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) placeHolder[i] = 0;
    for (int i = start; i < end; ++i) {
        placeHolder[i] = preact[i] + bias[i];
    }

    MPI_Allreduce(placeHolder, preact, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}


void bp_weight_f(float d_weight[10][6][6][6], const float d_preact[10], const float p_output[6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 10 * 6 * 6 * 6;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        int m = i / (6 * 6 * 6);
        int j = (i / (6 * 6)) % 6;
        int k = (i / 6) % 6;
        int l = i % 6;
        placeHolder[i] = d_preact[m] * p_output[j][k][l];
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    memcpy(d_weight, result, sizeof(float) * S);
}

void bp_bias_f(float bias[10], const float d_preact[10]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 10;
    float placeHolder[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        placeHolder[i] = bias[i] + dt * d_preact[i];
    }

    MPI_Allreduce(placeHolder, bias, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void bp_output_s1(float d_output[6][6][6], const float n_weight[10][6][6][6], const float nd_preact[10]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    float placeHolder[S];
    float result[S];

    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l] = 0;
    }

    it = ceil(10 * S/(float)p);
    start = r * it;
    end = Min(start + it, 10 * S);

    for (int i = start; i < end; ++i) {
        int i1 = i / (6 * 6 * 6);
        int i2 = (i / (6 * 6)) % 6;
        int i3 = (i / 6) % 6;
        int i4 = i % 6; 
        placeHolder[i % (6 * 6 * 6)] += n_weight[i1][i2][i3][i4] * nd_preact[i1];
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor666(d_output, result);
}


void bp_preact_s1(float d_preact[6][6][6], const float d_output[6][6][6], const float preact[6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    float placeHolder[S];
    float result[S];

    for (int l = 0; l < S; ++l) {
        placeHolder[l] = 0;
        result[l] = 0;
    }

    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int j = (l / 6) % 6;
        int k = l % 6;
        float o = step_function(preact[m][j][k]);
        placeHolder[l] = d_output[m][j][k] * o * (1 - o);;
    }

     MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor666(d_preact, result);
}

void bp_weight_s1(float d_weight[1][4][4], const float d_preact[6][6][6], const float p_output[6][24][24]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 4 * 4;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    float placeHolder[S];
    float result[S];

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }
    

    it = ceil(4 * 4 * 6 * 6 * 6/(float)p);
    start = r * it;
    end = Min(start + it, 4 * 4 * 6 * 6 * 6);

    for (int i = start; i < end; ++i) {
        int i2 = i / (4 * 6 * 6 * 6);
        int i3 = (i / (6 * 6 * 6)) % 4;
        int i4 = (i / (6 * 6)) % 6;
        int i5 = (i / 6) % 6;
        int i6 = i % 6;
        placeHolder[i2 * 4 + i3] += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3];
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor144(d_weight, result);
}

float bp_bias_s1(float bias[1], const float d_preact[6][6][6]) {
     int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    
    int S = 6 * 6 * 6;
    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    float placeHolder = 0.0;

    float sum = 0.0f;
    for (int l = start; l < end; ++l) {
        int m = l / (6 * 6);
        int j = (l / 6) % 6;
        int k = l % 6;
        sum += d_preact[m][j][k];
    }

    placeHolder = dt * sum / S;
    float gradient;
    MPI_Allreduce(&placeHolder, &gradient, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    bias[0] += gradient;
    return bias[0];
}

void bp_output_c1(float d_output[6][24][24], const float n_weight[1][4][4], const float nd_preact[6][6][6]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 24 * 24;
    float placeHolder[S];
    float result[S];

    int it = ceil(S/(float)p);
    int start = r * it;
    int end = Min(start + it, S);

    for (int i = 0; i < S; ++i) {
        placeHolder[i] = 0;
        result[i] = 0;
    }

    it = ceil(4 * 4 * 6 * 6 * 6/(float)p);
    start = r * it;
    end = Min(start + it, 4 * 4 * 6 * 6 * 6);

    for (int i = start; i < end; ++i) {
        int i1 = i / (4 * 4 * 6 * 6 * 6);
        int i2 = (i / (4 * 6 * 6 * 6)) % 4;
        int i3 = (i / (6 * 6 * 6)) % 4;
        int i4 = (i / (6 * 6)) % 6;
        int i5 = (i / 6) % 6;
        int i6 = i % 6;

        int x = i5 * 4 + i2;
        int y = i6 * 4 + i3;
        placeHolder[i4 * 6 * 24 + x * 24 + y] += n_weight[i1][i2][i3] * nd_preact[i4][i5][i6];
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor62424(d_output, result);
}

static float sigmoid_derivative(float x) {
    float s = step_function(x);
    return s * (1.0f - s);
}

void bp_preact_c1(float d_preact[6][24][24], const float d_output[6][24][24], const float preact[6][24][24]) {

     int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 24 * 24;
    float placeHolder[S];
    float result[S];

    int it = ceil(6 * 24 * 24/(float)p);
    int start = r * it;
    int end = Min(start + it, 6 * 24 * 24);

    for (int i =0 ; i < S; i++){
        placeHolder[i] = 0;
        result[i] = 0;
    }

    for (int i = start; i < end; ++i) {
        int m = i / (24 * 24);
        int j = (i / 24) % 24;
        int k = i % 24;
        placeHolder[i] = d_output[m][j][k] * sigmoid_derivative(preact[m][j][k]);
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor62424(d_preact, result);
}

void bp_weight_c1(float d_weight[6][5][5], const float d_preact[6][24][24], const float p_output[28][28]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    int S = 6 * 5 * 5;
    float placeHolder[S];
    float result[S];

    for (int i = 0; i < S; ++i) placeHolder[i] = result[i] = 0;

    int terms = 6 * 5 * 5 * 24 * 24;
    int it = ceil(terms / (float)p);
    int start = r * it, end = Min(start + it, terms);
    for (int i = start; i < end; ++i) {
        int y = i % 24;
        int x = (i / 24) % 24;
        int ky = (i / (24 * 24)) % 5;
        int kx = (i / (5 * 24 * 24)) % 5;
        int m = i / (25 * 24 * 24);
        placeHolder[m * 25 + kx * 5 + ky] += d_preact[m][x][y] * p_output[x + kx][y + ky] / (24.0f * 24.0f);
    }

    MPI_Allreduce(placeHolder, result, S, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    buildTensor655(d_weight, result);
}


void bp_bias_c1(float bias[6], const float d_preact[6][24][24]) {
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    float gradients[6] = {0}, total[6];
    int terms = 6 * 24 * 24;
    int it = ceil(terms / (float)p);
    int start = r * it, end = Min(start + it, terms);

    for (int i = start; i < end; ++i) {
        int m = i / (24 * 24);
        int pixel = i % (24 * 24);
        gradients[m] += d_preact[m][pixel / 24][pixel % 24];
    }
    MPI_Allreduce(gradients, total, 6, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < 6; ++i) bias[i] += dt * total[i] / (24.0f * 24.0f);
}

#endif /* LAYER_H */
