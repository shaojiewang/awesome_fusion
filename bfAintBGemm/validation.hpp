#pragma once
#include "datatype.hpp"

template <typename T>
static inline bool valid_vector( const float* ref, const T* pred, int n, float nrms = 1e-3 )
{    
    float s0 = 0.0;
    float s1 = 0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end = n;
    
    for(int i = i_start; i < i_end; ++i ){
        float ri = ref[i];
        float pi = type_convert<float>(pred[i]);
        float d = ri - pi;
        float dd = d * d;
        float rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        
#ifdef PER_PIXEL_CHECK
        float delta = ABS(ri - pi) / ri;
        if(delta > 1e-3){
#ifdef ASSERT_ON_FAIL
            if(pp_err < 100)
            printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n", i, ri, pi, ((uint16_t*)pred)[i], delta);
#endif
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err == 0)
#endif
    ;
}

