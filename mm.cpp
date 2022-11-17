#include <iostream>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
using namespace std;

#define RAND_LOWER_BOUND 1
#define RAND_UPPER_BOUND 2

#define SIZE_BOUND 256

/* Utils begin */
int32_t usage() {
    cout<<"\t./mm <N>"<<endl;
    return -1;
}

long long wall_clock_time()
{
#ifdef LINUX
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

void init_matrix(double *m, int r, int c) {
    double t = RAND_UPPER_BOUND - RAND_LOWER_BOUND;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            m[i*r+j] = (rand() / double(RAND_MAX)) * t + RAND_LOWER_BOUND;
        }
    }
}

void clear_matrix(double *m, int r, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            m[i*r+j] = 0;
        }
    }
}

void output_matrix(string info, double *m, int r, int c) {
    cout<<info<<endl;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            cout<<m[i*r+j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

bool check_correctness(int N, double *res1, double *res2) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(res1[i*N+j] - res2[i*N+j]) >= 0.01) {
				printf("i = %d, j = %d\n", i, j);
                return false;
            }
        }
    }
    return true;
}

/* Utils end */
void mm_seq_ijk(int N, double *A, double *B, double *C) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double sum = 0;
			for (int k = 0; k < N; ++k) {
				sum += A[i*N+k] * B[k*N+j];
			}
			C[i*N+j] = sum;
		}
	}
}


void mm_seq_ikj(int N, double *A, double *B, double *C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double t = A[i*N+k];
            for (int j = 0; j < N; ++j) {
                C[i*N+j] += t * B[k*N+j];
            }
        }
    }
}

void mm_parallel(int N, double *A, double *B, double *C) {
    int i, j, k;
#pragma omp parallel for private(i,j,k) shared(A, B, C)
    for (i = 0; i < N; ++i) {
        for (k = 0; k < N; ++k) {
            double t = A[i*N+k];
            for (j = 0; j < N; ++j) {
                C[i*N+j] += t * B[k*N+j];
            }
        }
    }
}

void matrix_sum_parallel(int N, double *A, double *B, double *C) {
    int i, j;
#pragma omp parallel for private(i,j) shared(A,B,C)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}

void matrix_sub_parallel(int N, double *A, double *B, double *C) {
    int i, j;
#pragma omp parallel for private(i,j) shared(A,B,C)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            C[i*N+j] = A[i*N+j] - B[i*N+j];
        }
    }
}

void matrix_divide_parallel(int N, double *A, int n, double *A11, double *A12, double *A21, double *A22) {
    int i, j;
#pragma omp parallel for private(i,j) shared(A,A11,A12,A21,A22)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            if (i < n && j < n) {
                A11[i*n+j] = A[i*N+j];
            } else if (i < n && j >= n) {
                A12[i*n+j-n] = A[i*N+j];
            } else if (i >= n && j < n) {
                A21[(i-n)*n+j] = A[i*N+j];
            } else {
                A22[(i-n)*n+j-n] = A[i*N+j];
            }
        }
    }
}

void matrix_merge_parallel(int N, double *C, int n, double *C11, double *C12, double *C21, double *C22) {
    int i, j;
#pragma omp parallel for private(i,j) shared(C,C11,C12,C21,C22)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            if (i < n && j < n) {
                C[i*N+j] = C11[i*n+j];
            } else if (i < n && j >= n) {
                C[i*N+j] = C12[i*n+j-n];
            } else if (i >= n && j < n) {
                C[i*N+j] = C21[(i-n)*n+j];
            } else if (i >= n && j >= n) {
                C[i*N+j] = C22[(i-n)*n+j-n];
            }
        }
    }
}

void mm_stranssen(int N, double *A, double *B, double *C) {
	if (N <= SIZE_BOUND) {
		mm_parallel(N, A, B, C);
		return;
	}

	int n = N / 2;

	double *A11 = (double*)malloc(n*n*sizeof(double));
	double *A12 = (double*)malloc(n*n*sizeof(double));
	double *A21 = (double*)malloc(n*n*sizeof(double));
	double *A22 = (double*)malloc(n*n*sizeof(double));
	matrix_divide_parallel(N, A, n, A11, A12, A21, A22);

	double *B11 = (double*)malloc(n*n*sizeof(double));
	double *B12 = (double*)malloc(n*n*sizeof(double));
	double *B21 = (double*)malloc(n*n*sizeof(double));
	double *B22 = (double*)malloc(n*n*sizeof(double));
	matrix_divide_parallel(N, B, n, B11, B12, B21, B22);

	double *S1 = (double*)malloc(n*n*sizeof(double));
	double *S2 = (double*)malloc(n*n*sizeof(double));
	double *S3 = (double*)malloc(n*n*sizeof(double));
	double *S4 = (double*)malloc(n*n*sizeof(double));
	double *S5 = (double*)malloc(n*n*sizeof(double));
	double *S6 = (double*)malloc(n*n*sizeof(double));
	double *S7 = (double*)malloc(n*n*sizeof(double));
	double *S8 = (double*)malloc(n*n*sizeof(double));
	double *S9 = (double*)malloc(n*n*sizeof(double));
	double *S10 = (double*)malloc(n*n*sizeof(double));

	double *P1 = (double*)malloc(n*n*sizeof(double));
	double *P2 = (double*)malloc(n*n*sizeof(double));
	double *P3 = (double*)malloc(n*n*sizeof(double));
	double *P4 = (double*)malloc(n*n*sizeof(double));
	double *P5 = (double*)malloc(n*n*sizeof(double));
	double *P6 = (double*)malloc(n*n*sizeof(double));
	double *P7 = (double*)malloc(n*n*sizeof(double));

#pragma omp task firstprivate(n) 
{
	matrix_sub_parallel(n, B12, B22, S1);
	mm_stranssen(n, A11, S1, P1);
}
#pragma omp task firstprivate(n)
{
	matrix_sum_parallel(n, A11, A12, S2);
	mm_stranssen(n, S2, B22, P2);
}
#pragma omp task firstprivate(n)
{
	matrix_sum_parallel(n, A21, A22, S3);
	mm_stranssen(n, S3, B11, P3);
}
#pragma omp task firstprivate(n)
{
	matrix_sub_parallel(n, B21, B11, S4);
	mm_stranssen(n, A22, S4, P4);
}
#pragma omp task firstprivate(n)
{
	matrix_sum_parallel(n, A11, A22, S5);
	matrix_sum_parallel(n, B11, B22, S6);
	mm_stranssen(n, S5, S6, P5);
}
#pragma omp task firstprivate(n)
{
	matrix_sub_parallel(n, A12, A22, S7);
	matrix_sum_parallel(n, B21, B22, S8);
	mm_stranssen(n, S7, S8, P6);
}
#pragma omp task firstprivate(n)
{
	matrix_sub_parallel(n, A11, A21, S9);
	matrix_sum_parallel(n, B11, B12, S10);
	mm_stranssen(n, S9, S10, P7);
}
#pragma omp taskwait
	double *C11 = (double*)malloc(n*n*sizeof(double));
	matrix_sum_parallel(n, P4, P5, C11);
	matrix_sum_parallel(n, C11, P6, C11);
	matrix_sub_parallel(n, C11, P2, C11);
	
	double *C12 = (double*)malloc(n*n*sizeof(double));
	matrix_sum_parallel(n, P1, P2, C12);
	
	double *C21 = (double*)malloc(n*n*sizeof(double));
	matrix_sum_parallel(n, P3, P4, C21);
	
	double *C22 = (double*)malloc(n*n*sizeof(double));
	matrix_sum_parallel(n, P1, P5, C22);
	matrix_sub_parallel(n, C22, P3, C22);
	matrix_sub_parallel(n, C22, P7, C22);
	
	matrix_merge_parallel(N, C, n, C11, C12, C21, C22);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        return usage();
    }

	long long before, after;

    uint32_t N   = atoi(argv[1]);

	double *A = (double*)malloc(N*N*sizeof(double));
	double *B = (double*)malloc(N*N*sizeof(double));
	init_matrix(A, N, N);
	init_matrix(B, N, N);

	double *C_seq1 = (double*)malloc(N*N*sizeof(double));
	before = wall_clock_time();
	mm_seq_ijk(N, A, B, C_seq1);
	after = wall_clock_time();
	printf("mm_seq_ijk finished in %f s\n",((float)(after - before)) / 1000000000);
	
	double *C_seq2 = (double*)malloc(N*N*sizeof(double));
	before = wall_clock_time();
	mm_seq_ikj(N, A, B, C_seq2);
	after = wall_clock_time();
	printf("mm_seq_ikj finished in %f s\n",((float)(after - before)) / 1000000000);

	double *C_para = (double*)malloc(N*N*sizeof(double));
	before = wall_clock_time();
	mm_parallel(N, A, B, C_para);
	after = wall_clock_time();
	printf("mm_parallel finished in %f s\n",((float)(after - before)) / 1000000000);

	double *C_stranssen = (double*)malloc(N*N*sizeof(double));
	before = wall_clock_time();
	mm_stranssen(N, A, B, C_stranssen);
	after = wall_clock_time();
	printf("mm_stranssen finished in %f s\n",((float)(after - before)) / 1000000000);
	
	if (!check_correctness(N, C_seq1, C_seq2)) {
		printf("C_seq1 result error\n");
		return -1;
	}

	if (!check_correctness(N, C_para, C_seq2)) {
		printf("C_para result error\n");
		return -1;
	}
	if (!check_correctness(N, C_stranssen, C_seq2)) {
		printf("C_stranssen result error\n");
		return -1;
	}

	// test divide and merge
	/*int n = N/2;
	double *A11 = (double*)malloc(n*n*sizeof(double));
	double *A12 = (double*)malloc(n*n*sizeof(double));
	double *A21 = (double*)malloc(n*n*sizeof(double));
	double *A22 = (double*)malloc(n*n*sizeof(double));
	matrix_divide_parallel(N, A, n, A11, A12, A21, A22);

	double *A2 = (double*)malloc(N*N*sizeof(double));
	matrix_merge_parallel(N, A2, n, A11, A12, A21, A22);
	output_matrix("matrix A:", A, N, N);
	output_matrix("matrix A2:", A2, N, N);

	if (!check_correctness(N, A, A2)) {
		printf("result error\n");
		return -1;
	}*/

	return 0;
}
