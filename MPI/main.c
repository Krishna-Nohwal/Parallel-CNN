#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static Layer l_input, l_c1, l_s1, l_f;

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

static float vector_norm(const float *vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(void) {
    MPI_Init(NULL, NULL);
    srand(time(NULL));
    layer_init(&l_input, 0, 0, 28*28);
    layer_init(&l_c1, 5*5, 6, 24*24*6);
    layer_init(&l_s1, 4*4, 1, 6*6*6);
    layer_init(&l_f, 6*6*6, 10, 10);

    loaddata();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) fprintf(stdout, "Dataset: %u training images, %u test images\n", train_cnt, test_cnt);
    learn();
    test();

    layer_destroy(&l_input);
    layer_destroy(&l_c1);
    layer_destroy(&l_s1);
    layer_destroy(&l_f);
    MPI_Finalize();
    return 0;
}

static double forward_pass(double data[28][28]) {
  float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

    layer_clear(&l_input);
    layer_clear(&l_c1);
    layer_clear(&l_s1);
    layer_clear(&l_f);

	clock_t start, end;
	start = clock();

    layer_set_output(&l_input, (float *)input);
	

    fp_preact_c1((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
    fp_bias_c1((float (*)[24][24])l_c1.preact, l_c1.bias);
    apply_step_function(l_c1.preact, l_c1.output, l_c1.O);

    fp_preact_s1((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
    fp_bias_s1((float (*)[6][6])l_s1.preact, l_s1.bias);
    apply_step_function(l_s1.preact, l_s1.output, l_s1.O);

    fp_preact_f((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
    fp_bias_f(l_f.preact, l_f.bias);
    apply_step_function(l_f.preact, l_f.output, l_f.O);

    end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double back_pass() {
    clock_t start = clock();

    bp_weight_f((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
    bp_bias_f(l_f.bias, l_f.d_preact);

    bp_output_s1((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
    bp_preact_s1((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
    bp_weight_s1((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
    bp_bias_s1(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

    bp_output_c1((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
    bp_preact_c1((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
    bp_weight_c1((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
    bp_bias_c1(l_c1.bias, (float (*)[24][24])l_c1.d_preact);

	apply_grad(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static void learn() {
    float err;
    int iter = 1;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    unsigned int progress_step = train_cnt / 20;
    if (progress_step == 0) progress_step = 1;
    double time_taken = 0.0;

    if (rank == 0) {
        fprintf(stdout, "Learning\n");
        fprintf(stdout, "Training images: %u | batch size: 1 | batches per epoch: %u\n", train_cnt, train_cnt);
    }

    while (iter < 0 || iter-- > 0) {
        err = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            time_taken += forward_pass(train_set[i].data);

            layer_bp_clear(&l_f);
            layer_bp_clear(&l_s1);
            layer_bp_clear(&l_c1);

            // Euclid distance of train_set[i]
    makeError(l_f.d_preact, l_f.output, train_set[i].label, 10);
            tmp_err = vector_norm(l_f.d_preact, 10);
            err += tmp_err;
            time_taken += back_pass();
            unsigned int done = (unsigned int)i + 1;
            if (rank == 0 && (done % progress_step == 0 || done == train_cnt))
                fprintf(stdout, "Epoch 1/1: %u/%u batches (%u%%)\n", done, train_cnt, done * 100 / train_cnt);
        }

        err /= train_cnt;
        if (rank == 0) fprintf(stdout, "error: %e, time_on_cpu: %lf\n", err, time_taken);

        if (err < threshold) {
            if (rank == 0) fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }
    }

    if (rank == 0) fprintf(stdout, "\nTime - %lf\n", time_taken);
}

static unsigned int classify(double data[28][28]) {
    forward_pass(data);
    unsigned int max = 0;
    for (int i = 1; i < 10; ++i) {
        if (l_f.output[max] < l_f.output[i]) {
            max = i;
        }
    }

    return max;
}

static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		(double)error / (double)test_cnt * 100.0);
}
