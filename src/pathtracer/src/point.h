#ifndef POINT_H
#define POINT_H

#include "cudadefines.h"
#include "vector.h"

template<typename T>
struct Point4;

template<typename T>
struct Point3{
   T x,y,z;
   CPU_GPU Point3();
   CPU_GPU Point3(T, T, T);
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
   CPU_GPU Point3<T>& operator*=(const T s);
   CPU_GPU Point3<T> operator*(const T s) const;
};

template<typename T>
CPU_GPU inline Point3<T> operator*(const T s, const Point3<T>& v);


template<typename T>
struct Point4{
   T x,y,z,w;
   CPU_GPU Point4();
   CPU_GPU Point4(T, T, T, T);
   CPU_GPU explicit operator Point3<T>() const;
};

typedef Point3<float> Point3f;

#endif