# Optimization of Matrix Multiplication

System: Linux Ubuntu 20.04 

CPU: 4 Intel(R) Core(TM) i7-7560U CPU @ 2.40GHz



* Baseline

```c++
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
```

It takes about **18.17s** to finish `./mm 1024`



* Utilize cache more efficiently,

```c++
// A: row-wise read
// B: row-wise read
// C: row-wise write
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
```

In this way, there will be no column-wise read/write, the running time of `./mm 1024` decreases to about **3.05s**



* Utlize compiler options

```shell
g++ -O3 mm.cpp -o mm
```

Use the compiler to optimize the running time. Now the running time of `./mm 1024` decrease to about **0.52s**



* Do in parallel

```c++
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
```

With OpenMP we make better use of the computation power of the machine, now the running time decrease to about **0.19s**



* Change Algorithm to make it even faster

We make use of [Strassen algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm), which can reduce the time complexity of matrix multiplication from O(n^3) to around O(n^2.8) by **divide and conquer**.

After applying this method, the runinng time decrease to about **0.16s !**



|      | mm_seq_ijk -O0 | mm_seq_ikj -O0 | mm_parallel -O0 | mm_stranssen -O0 | mm_parallel -O3 | mm_stranssen -O3 |
| ---- | -------------- | -------------- | --------------- | ---------------- | --------------- | ---------------- |
| 32   | 0.000388       | 0.000196       | 0.000150        |                  | 0.000377        |                  |
| 64   | 0.000793       | 0.000841       | 0.000530        |                  | 0.000187        |                  |
| 128  | 0.006461       | 0.006135       | 0.003094        |                  | 0.000542        |                  |
| 256  | 0.066361       | 0.045085       | 0.023403        | 0.023253         | 0.003180        | 0.002994         |
| 512  | 1.244162       | 0.364017       | 0.189130        | 0.170443         | 0.020252        | 0.022698         |
| 1024 | 20.593155      | 3.045058       | 1.515647        | 1.197208         | 0.188845        | 0.165639         |
| 2048 |                | 24.553389      | 14.192327       | 10.471990        | 1.586314        | 1.210377         |

**![img](https://lh4.googleusercontent.com/cykEfqhMapFifYTIxWhvzpbbvk_94-ehXzbG26riGaBQKh5xuQ4-aHzffBQ6Sy5cE-t4QgPm7BDsZ0OPATQSsWz7ut7CnVUgVrH_dkpM-PzeGZE3J72uFhPLXlUCjZn2jqNqSdP9mH3MIuUdLrIo9qWShyE_FvPGmNj6S10-Sw6-GatpYcbv76Pm3b46)**

![img](https://lh5.googleusercontent.com/Z-oNyhnbONHXQH6hXvAF_DConn3i5V8nqfcDk8WGT-WXBa9_I6EZbP_bW_NPGdeZisSw01NhfmXGPgzyf31ghuRSQNXIzZveSia_pvZQLHL2h8AlevSoJzDEdBIsbRDdE3o0ai_oOgIKiPuwUV9CH41f3oGBhy7iWKBll89FG1PItHNzzOKIGf_uJZxk)

