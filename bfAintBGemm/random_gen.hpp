#pragma once

#include <random>

static inline void rand_vector_2d(float* v, int row, int col, int ld, float min_v = 0, float max_v = 1){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
            // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
        }
    }
}

static inline void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(std::rand() % 10)) - 5;
        }
    }
}

