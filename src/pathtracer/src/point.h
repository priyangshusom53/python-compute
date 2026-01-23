#ifndef POINT_H
#define POINT_H

#include "cudadefines.h"
#include "vector.h"

#include<cmath>

template<typename T>
struct Point4;

template<typename T>
struct Point3{
   T x,y,z;
   CPU_GPU Point3();
   CPU_GPU Point3(T);
   CPU_GPU Point3(T, T, T);
   CPU_GPU T& operator[](int);
   CPU_GPU T  operator[](int) const;
   CPU_GPU explicit Point3(const Point4<T>&);
   CPU_GPU explicit operator Point4<T>() const;
   CPU_GPU Point3<T>& operator+=(const Vector3<T>& v);
   CPU_GPU Point3<T> operator+(const Vector3<T> &v) const;
   CPU_GPU Point3<T>& operator-=(const Vector3<T>& v);
   CPU_GPU Point3<T> operator-(const Vector3<T> &v) const;
   CPU_GPU Vector3<T> operator-(const Point3<T> &p) const;
   CPU_GPU static T Distance(const Point3<T> &p1, const Point3<T> &p2);
   CPU_GPU Point3<T>& operator+=(const Point3<T>& other);
   CPU_GPU Point3<T> operator+(const Point3<T>& other) const;
   CPU_GPU Point3<T>& operator*=(T s);
   CPU_GPU Point3<T> operator*(T s) const;
   CPU_GPU static Point3<T> Lerp(float t, const Point3<T> &p0, const Point3<T> &p1);
   CPU_GPU static Point3<T> Min(const Point3<T> &v1, const Point3<T> &v2);
   CPU_GPU static Point3<T> Max(const Point3<T> &v1, const Point3<T> &v2);
   CPU_GPU static Point3<T> Permute(const Point3<T> &v, int x, int y, int z);
};

template<typename T>
CPU_GPU inline Point3<T> operator*(T s, const Point3<T>& v);


template<typename T>
struct Point4{
   T x,y,z,w;
   CPU_GPU Point4();
   CPU_GPU Point4(T, T, T, T);
   CPU_GPU explicit operator Point3<T>() const;
};

template<typename T>
struct Point2{
   T x,y;
   CPU_GPU Point2();
   CPU_GPU Point2(T);
   CPU_GPU Point2(T, T);
};

typedef Point3<float> Point3f;
typedef Point4<float> Point4f;
typedef Point2<float> Point2f;

// Point3 definition
template<typename T>
CPU_GPU Point3<T>::Point3() : x(0), y(0), z(0) {}

template<typename T>
CPU_GPU Point3<T>::Point3(T _x) : x(_x), y(_x), z(_x) {}

template<typename T>
CPU_GPU Point3<T>::Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

template<typename T>
CPU_GPU T& Point3<T>::operator[](int idx) {
   return (&x)[idx];
}

template<typename T>
CPU_GPU T Point3<T>::operator[](int idx) const {
   return (&x)[idx];
}

template<typename T>
CPU_GPU Point3<T>::Point3(const Point4<T>& p) : x(p.x), y(p.y), z(p.z) {}

template<typename T>
CPU_GPU Point3<T>::operator Point4<T>() const {
   return Point4<T>(x, y, z, 1);
}

template<typename T>
CPU_GPU Point3<T>& Point3<T>::operator+=(const Vector3<T>& v) {
   this->x += v.x;
   this->y += v.y;
   this->z += v.z;
   return *this;
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::operator+(const Vector3<T>& v) const {
   Point3<T> result = *this;
   return result += v;
}

template<typename T>
CPU_GPU Point3<T>& Point3<T>::operator-=(const Vector3<T>& v) {
   this->x -= v.x;
   this->y -= v.y;
   this->z -= v.z;
   return *this;
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::operator-(const Vector3<T>& v) const {
   Point3<T> result = *this;
   return result -= v;
}

template<typename T>
CPU_GPU Vector3<T> Point3<T>::operator-(const Point3<T>& p) const {
   return Vector3<T>(x-p.x, y-p.y, z-p.z);
}

template<typename T>
CPU_GPU T Point3<T>::Distance(const Point3<T>& p1, const Point3<T>& p2) {
   return (p1 - p2).Length();
}

template<typename T>
CPU_GPU Point3<T>& Point3<T>::operator+=(const Point3<T>& other) {
   this->x += other.x;
   this->y += other.y;
   this->z += other.z;
   return *this;
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::operator+(const Point3<T>& other) const {
   Point3<T> result = *this;
   return result += other;
}

template<typename T>
CPU_GPU Point3<T>& Point3<T>::operator*=(T s) {
   this->x *= s;
   this->y *= s;
   this->z *= s;
   return *this;
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::operator*(T s) const {
   Point3<T> result = *this;
   return result *= s;
}

template<typename T>
CPU_GPU Point3<T> operator*(T s, const Point3<T>& p){
   Point3<T> result = p * s;
   return result;
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::Lerp(float t, const Point3<T>& p0, const Point3<T>& p1) {
   return (1-t) * p0 + t * p1;
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::Min(const Point3<T>& p1, const Point3<T>& p2) {
#ifdef __CUDA_ARCH__

   return Point3<T>(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
#else
   return Point3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
#endif
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::Max(const Point3<T>& p1, const Point3<T>& p2) {

#ifdef __CUDA_ARCH__
   return Point3<T>(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
#else
   return Point3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),std::max(p1.z, p2.z));
#endif
}

template<typename T>
CPU_GPU Point3<T> Point3<T>::Permute(const Point3<T>& p, int x, int y, int z) {
   return Point3<T>(p[x], p[y], p[z]);
}

// Point4 definition
template<typename T>
CPU_GPU Point4<T>::Point4() : x(0), y(0), z(0), w(1) {}

template<typename T>
CPU_GPU Point4<T>::Point4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}

template<typename T>
CPU_GPU Point4<T>::operator Point3<T>() const{
   return Point3<T>(x,y,z);
}


// Point2 definition
template<typename T>
CPU_GPU Point2<T>::Point2() : x(0), y(0) {}

template<typename T>
CPU_GPU Point2<T>::Point2(T _x): x(_x), y(_x) {}

template<typename T>
CPU_GPU Point2<T>::Point2(T _x, T _y) : x(_x), y(_y) {}

#endif