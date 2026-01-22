#ifndef NORMAL_H
#define NORMAL_H

#include "cudadefines.h"
#include "vector.h"

#include<algorithm>
#include<cmath>

template<typename T>
struct Normal3
{
   T x,y,z;
   CPU_GPU Normal3();
   CPU_GPU Normal3(T _x, T _y, T _z);
   CPU_GPU explicit Normal3(const Vector3<T> &v);
   CPU_GPU explicit operator Vector3<T>() const;
   CPU_GPU T& operator[](int);
   CPU_GPU T  operator[](int) const;
   CPU_GPU Normal3<T>& operator+=(const Normal3<T> &other);
   CPU_GPU Normal3<T> operator+(const Normal3<T> &other) const;
   CPU_GPU Normal3<T>& operator-=(const Normal3<T> &other);
   CPU_GPU Normal3<T> operator-(const Normal3<T> &other) const;
   CPU_GPU Normal3<T>& operator*=(T s);
   CPU_GPU Normal3<T> operator*(T s) const;
   CPU_GPU static T Dot(const Normal3<T> &n1, const Normal3<T> &n2);
   CPU_GPU static T Dot(const Normal3<T> &n, const Vector3<T> &v);
   CPU_GPU static T Dot(const Vector3<T>& v, const Normal3<T>& n);
   CPU_GPU static T AbsDot(const Normal3<T> &n1, const Normal3<T> &n2);
   CPU_GPU static T AbsDot(const Normal3<T> &n, const Vector3<T> &v);
   CPU_GPU static Normal3<T> Normalize(const Normal3<T> &n);
   CPU_GPU static Normal3<T> FaceForward(const Normal3<T> &n, const Vector3<T> &v);
};

template<typename T>
CPU_GPU inline Normal3<T> operator*(T s, const Normal3<T> &n);

typedef Normal3<float> Normal3f;


// Normal3 definition
template<typename T>
CPU_GPU Normal3<T>::Normal3() : x(0), y(0), z(0) {}

template<typename T>
CPU_GPU Normal3<T>::Normal3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

template<typename T>
CPU_GPU Normal3<T>::Normal3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}

template<typename T>
CPU_GPU Normal3<T>::operator Vector3<T>()const {
	return Vector3<T>(x, y, z);
}

template<typename T>
CPU_GPU T& Normal3<T>::operator[](int idx) {
   return (&x)[idx];
}

template<typename T>
CPU_GPU T  Normal3<T>::operator[](int idx) const{
   return (&x)[idx];
}

template<typename T>
CPU_GPU Normal3<T>& Normal3<T>::operator+=(const Normal3<T> &other){
   this->x+=other.x;
   this->y+=other.y;
   this->z+=other.z;
   return *this;
}

template<typename T>
CPU_GPU Normal3<T> Normal3<T>::operator+(const Normal3<T> &other) const{
   Normal3<T> result = *this;
   return result += other;
}

template<typename T>
CPU_GPU Normal3<T>& Normal3<T>::operator-=(const Normal3<T> &other){
   this->x-=other.x;
   this->y-=other.y;
   this->z-=other.z;
   return *this;
}

template<typename T>
CPU_GPU Normal3<T> Normal3<T>::operator-(const Normal3<T> &other) const{
   Normal3<T> result = *this;
   return result -= other;
}

template<typename T>
CPU_GPU Normal3<T>& Normal3<T>::operator*=(const T s){
   this->x*=s;
   this->y*=s;
   this->z*=s;
   return *this;
}

template<typename T>
CPU_GPU Normal3<T> Normal3<T>::operator*(const T s) const{
   Normal3<T> result = *this;
   return result *= s;
}

template<typename T>
CPU_GPU Normal3<T> operator*(const T s, const Normal3<T> &n){
   return n * s;
}

template<typename T>
CPU_GPU T Normal3<T>::Dot(const Normal3<T> &n1, const Normal3<T> &n2){
   return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

template<typename T>
CPU_GPU T Normal3<T>::Dot(const Normal3<T> &n, const Vector3<T> &v){
   return n.x * v.x + n.y * v.y + n.z * v.z;
}

template<typename T>
CPU_GPU T Dot(const Vector3<T>& v, const Normal3<T>& n) {
	return Normal3<T>::Dot(n, v);
}

template<typename T>
CPU_GPU T Normal3<T>::AbsDot(const Normal3<T> &n1, const Normal3<T> &n2){
#ifdef __CUDA_ARCH__
	return abs(Normal3<T>::Dot(n1, n2));
#else
	return std::abs(Normal3<T>::Dot(n1, n2));
#endif
}

template<typename T>
CPU_GPU T Normal3<T>::AbsDot(const Normal3<T> &n, const Vector3<T> &v){
#ifdef __CUDA_ARCH__
	return abs(Normal3<T>::Dot(n, v));
#else
	return std::abs(Normal3<T>::Dot(n, v));
#endif
}

template<typename T>
CPU_GPU Normal3<T> Normal3<T>::Normalize(const Normal3<T> &n){
#ifdef __CUDA_ARCH__
   T invL = T(1)/sqrt(Dot(n,n));
   return n * invL;
#else
   T invL = T(1)/std::sqrt(Dot(n,n));
   return n * invL;
#endif
}

template<typename T>
CPU_GPU Normal3<T> Normal3<T>::FaceForward(const Normal3<T> &n, const Vector3<T> &v){
   return n * Dot(n,v) * (1/AbsDot(n,v));
}

#endif