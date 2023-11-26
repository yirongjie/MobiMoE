// hello.cpp
#include <stdio.h>
#include <omp.h> // 包含 openMP用到的库
#include <time.h>
#include <stdio.h>
#include <chrono>

int main(void)
{
    float centroids[] = { -0.23291062, -0.15733099, -0.11985998, -0.09184306, -0.06838582, -0.04740785,
                        -0.02787124, -0.00910846,  0.0097356,  0.02842614,  0.04804431,  0.06912179,
                        0.09273144,  0.12050226,  0.15781654,  0.23348868};
    
    // #pragma omp parallel //到括号终止处的部分将会多线程执行
    // printf("Hello, world.\n");
    // i = centroids[quantize_[1]]
    float result[100];

    // clock_t start = clock();
    auto begin = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
	for (int i = 0; i < 2359296; i++)
		// printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
        result[0] = centroids[1];

    // Stop measuring time and calculate the elapsed time
    // clock_t end = clock();
    // double elapsed = double(end - start)/CLOCKS_PER_SEC;
    
    // printf("Time measured: %.3f seconds.\n", elapsed);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    
    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

    return 0;
}
