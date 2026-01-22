#ifndef UTILS_H
#define UTILS_H

#include "cudadefines.h"

CPU_GPU float Lerp(float t, const float &p1, const float &p2){
   return (1-t) * p1 + t * p2;
}


#endif