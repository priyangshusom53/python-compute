#ifndef VECTOR_H
#define VECTOR_H

#include"cudadefines.h"

template<typename T>
struct Vector3
{
   T x,y,z;
   CPU_GPU Vector3();
   CPU_GPU Vector3(T _x, T _y, T _z);
   CPU_GPU Vector3(const Vector4<T> &);
   CPU_GPU T operator[](const int);
   CPU_GPU Vector3<T>& operator+=(const Vector3<T>& other);
   CPU_GPU Vector3<T> operator+(const Vector3<T>& other) const;
   CPU_GPU Vector3<T>& operator-=(const Vector3<T> other);
   CPU_GPU Vector3<T> operator-(const Vector3<T> other) const;
   CPU_GPU Vector3<T>& operator*=(const T s);
   CPU_GPU Vector3<T> operator*(const T s, const Vector3<T>& v) const;
   CPU_GPU Vector3<T> operator*(const Vector3<T>& v, const T s) const;
   CPU_GPU Vector3<T>& operator/=(const T invS);
   CPU_GPU Vector3<T> operator/(const Vector3<T>& v, const T invS) const;
   CPU_GPU T LengthSquared();
   CPU_GPU T Length();
   CPU_GPU Vector3<T> Normalize(const Vector3<T>&);
   CPU_GPU T Dot(const Vector3<T> &v1, const Vector3<T> &v2) const;
   CPU_GPU T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) const;
   CPU_GPU Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) const;
   CPU_GPU T MinComponent(const Vector3<T>&);
   CPU_GPU T MaxComponent(const Vector3<T>&);
   CPU_GPU int MaxDimension(const Vector3<T> &);
   CPU_GPU Vector3<T> Min(const Vector3<T> &p1, const Vector3<T> &p2);
   CPU_GPU Vector3<T> Max(const Vector3<T> &p1, const Vector3<T> &p2);
   CPU_GPU Vector3<T> Permute(const Vector3<T> &v, int x, int y, int z);
};

template<typename T>
struct Vector4
{
   T x,y,z,w;
   CPU_GPU Vector4();
   CPU_GPU Vector4(T _x, T _y, T _z, T _w);
   CPU_GPU Vector4(const Vector3<T>&, T);
   // CPU_GPU Vector4<T>& operator+=(const Vector4<T>& other);
   // CPU_GPU Vector4<T> operator+(const Vector4<T>& other) const;
};

using Vec4f = Vector4<float>;
using Vec3f = Vector3<float>;

#endif