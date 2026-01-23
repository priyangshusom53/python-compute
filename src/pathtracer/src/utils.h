#ifndef UTILS_H
#define UTILS_H

#include "cudadefines.h"

template<typename T>
CPU_GPU T Sqr(T n) {
	return n * n;
}

template<typename T>
CPU_GPU T Pow(T n, int ex) {
	T _n = T(1);
	for (int i = 0; i < ex; ++i) {
		_n *= n;
	}
	return _n;
}

CPU_GPU float Lerp(float t, const float &p1, const float &p2){
   return (1-t) * p1 + t * p2;
}


#endif