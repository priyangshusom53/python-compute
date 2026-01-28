#ifndef VECTOR_H
#define VECTOR_H

#include"cudadefines.h"

#include<algorithm>
#include<cmath>


template<typename T>
struct Vector4;

template<typename T>
struct Vector2;

template<typename T>
struct Vector3
{
   T x,y,z;
   CPU_GPU Vector3();
   CPU_GPU Vector3(T _x);
   CPU_GPU Vector3(T _x, T _y, T _z);
   CPU_GPU explicit Vector3(const Vector4<T> &);
   CPU_GPU explicit operator Vector4<T>() const;
   CPU_GPU explicit operator Vector2<T>() const; 
   CPU_GPU T& operator[](int);
   CPU_GPU T  operator[](int) const;
   CPU_GPU Vector3<T>& operator+=(const Vector3<T>& other);
   CPU_GPU Vector3<T> operator+(const Vector3<T>& other) const;
   CPU_GPU Vector3<T>& operator-=(const Vector3<T>& other);
   CPU_GPU Vector3<T> operator-(const Vector3<T>& other) const;
   CPU_GPU Vector3<T>& operator*=(T s);
   CPU_GPU Vector3<T> operator*(T s) const;
   CPU_GPU Vector3<T>& operator/=(T invS);
   CPU_GPU Vector3<T> operator/(T invS) const;
   CPU_GPU T LengthSquared() const;
   CPU_GPU T Length() const;
   CPU_GPU static Vector3<T> Normalize(const Vector3<T>&);
   CPU_GPU static T Dot(const Vector3<T> &v1, const Vector3<T> &v2);
   CPU_GPU static T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2);
   CPU_GPU static Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2);
   CPU_GPU static T MinComponent(const Vector3<T>&);
   CPU_GPU static T MaxComponent(const Vector3<T>&);
   CPU_GPU static int MaxDimension(const Vector3<T> &);
   CPU_GPU static Vector3<T> Min(const Vector3<T> &v1, const Vector3<T> &v2);
   CPU_GPU static Vector3<T> Max(const Vector3<T> &v1, const Vector3<T> &v2);
   CPU_GPU static Vector3<T> Permute(const Vector3<T> &v, int x, int y, int z);
   CPU_GPU static void CoordinateSystem(const Vector3<T> &v1,Vector3<T> *v2, Vector3<T> *v3);
};

template<typename T>
CPU_GPU inline Vector3<T> operator*(T s, const Vector3<T>& v);


template<typename T>
struct Vector4
{
   T x,y,z,w;
   CPU_GPU Vector4();
   CPU_GPU Vector4(T _x, T _y, T _z, T _w);
   CPU_GPU explicit Vector4(const Vector3<T>&);
   CPU_GPU Vector4(const Vector3<T>&, T);
   CPU_GPU explicit operator Vector3<T>() const;
   // CPU_GPU Vector4<T>& operator+=(const Vector4<T>& other);
   // CPU_GPU Vector4<T> operator+(const Vector4<T>& other) const;
};

template<typename T>
struct Vector2{
   T x,y;
   CPU_GPU Vector2();
   CPU_GPU Vector2(T _x, T _y);
};

typedef Vector4<float> Vector4f;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
typedef Vector2<float> Vector2f;


// Vector3 definition
template<typename T>
CPU_GPU Vector3<T>::Vector3() : x(0),y(0),z(0){}

template<typename T>
CPU_GPU Vector3<T>::Vector3(T _x) : x(_x), y(_x), z(_x) {}

template<typename T>
CPU_GPU Vector3<T>::Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

template<typename T>
CPU_GPU Vector3<T>::Vector3(const Vector4<T> &v) : x(v.x), y(v.y), z(v.z) {}

template<typename T>
CPU_GPU Vector3<T>::operator Vector4<T>() const {
   return Vector4<T>(x,y,z,0);
}

template<typename T>
CPU_GPU Vector3<T>::operator Vector2<T>() const {
   return Vector2<T>(x,y);
}

template<typename T>
CPU_GPU T& Vector3<T>::operator[](int idx){
   return (&x)[idx];
}

template<typename T>
CPU_GPU T Vector3<T>::operator[](int idx) const {
   return (&x)[idx];
}

template<typename T>
CPU_GPU Vector3<T>& Vector3<T>::operator+=(const Vector3<T>& other){
   this->x+=other.x;
   this->y+=other.y;
   this->z+=other.z;
   return *this;
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::operator+(const Vector3<T>& other) const {
   Vector3<T> result = *this;
   return result+=other;
}

template<typename T>
CPU_GPU Vector3<T>& Vector3<T>::operator-=(const Vector3<T> &other) {
   this->x -= other.x;
   this->y -= other.y;
   this->z -= other.z;
   return *this;
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::operator-(const Vector3<T> &other) const {
   Vector3<T> result = *this;
   return result -= other;
}

template<typename T>
CPU_GPU Vector3<T>& Vector3<T>::operator*=(T s){
   this->x *= s;
   this->y *= s;
   this->z *= s;
   return *this;
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::operator*(T s) const {
   Vector3<T> result = *this;
   return result *= s;
}

template<typename T>
CPU_GPU Vector3<T> operator*(T s, const Vector3<T>& v){
   Vector3<T> result = v * s;
   return result;
}

template<typename T>
CPU_GPU Vector3<T>& Vector3<T>::operator/=(T s){
   this->x /= s;
   this->y /= s;
   this->z /= s;
   return *this;
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::operator/(T s) const {
   Vector3<T> result = *this;
   return result /= s;
}

template<typename T>
CPU_GPU inline T Vector3<T>::LengthSquared() const {
   return x * x + y * y + z * z;
}

template<typename T>
CPU_GPU inline T Vector3<T>::Length() const {
   
#ifdef __CUDA_ARCH__
	return sqrt(this->LengthSquared());
#else
	return std::sqrt(this->LengthSquared());
#endif
}

template<typename T>
CPU_GPU inline Vector3<T> Vector3<T>::Normalize(const Vector3<T> &v){
   T invL = T(1)/v.Length();
   return v * invL;
}

template<typename T>
CPU_GPU T Vector3<T>::Dot(const Vector3<T> &v1, const Vector3<T> &v2){
   return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T>
CPU_GPU T Vector3<T>::AbsDot(const Vector3<T> &v1, const Vector3<T> &v2){
   
#ifdef __CUDA_ARCH__
	return abs(Vector3<T>::Dot(v1, v2));
#else
	return std::abs(Vector3<T>::Dot(v1, v2));
#endif
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::Cross(const Vector3<T> &v1, const Vector3<T> &v2){
    return Vector3<T>(
        (v1.y * v2.z) - (v1.z * v2.y),
        (v1.z * v2.x) - (v1.x * v2.z),
        (v1.x * v2.y) - (v1.y * v2.x)
    );
}

template<typename T>
CPU_GPU T Vector3<T>::MinComponent(const Vector3<T> &v){

#ifdef __CUDA_ARCH__
   return min(min(v.x, v.y), v.z);
#else
   return std::min(v.x, std::min(v.y, v.z));
#endif
}

template<typename T>
CPU_GPU T Vector3<T>::MaxComponent(const Vector3<T> &v){

#ifdef __CUDA_ARCH__
   return max(max(v.x, v.y), v.z);
#else
   return std::max(v.x, std::max(v.y, v.z));
#endif
}

template<typename T>
CPU_GPU int Vector3<T>::MaxDimension(const Vector3<T> &v){

   return (v.x > v.y) ? ((v.x > v.z) ?0:2) : ((v.y > v.z) ? 1 : 2);
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::Min(const Vector3<T> &v1, const Vector3<T> &v2){

#ifdef __CUDA_ARCH__

   return Vector3<T>(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z));
#else
   return Vector3<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
#endif
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::Max(const Vector3<T> &v1, const Vector3<T> &v2){

#ifdef __CUDA_ARCH__
   return Vector3<T>(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z));
#else
   return Vector3<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y),std::max(v1.z, v2.z));
#endif
}

template<typename T>
CPU_GPU Vector3<T> Vector3<T>::Permute(const Vector3<T> &v, int x, int y, int z){

   return Vector3<T>(v[x], v[y], v[z]);
}

template<typename T>
CPU_GPU void Vector3<T>::CoordinateSystem(const Vector3<T> &v1,Vector3<T> *v2, Vector3<T> *v3){

#ifdef __CUDA_ARCH__
   if (abs(v1.x) > abs(v1.y))
      *v2 = Vector3<T>(-v1.z, 0, v1.x) /
            sqrt(v1.x * v1.x + v1.z * v1.z);
   else
      *v2 = Vector3<T>(0, v1.z,-v1.y) /
            sqrt(v1.y * v1.y + v1.z * v1.z);
   *v3 = Cross(v1, *v2);
#else
   if (std::abs(v1.x) > std::abs(v1.y))
      *v2 = Vector3<T>(-v1.z, 0, v1.x) /
            std::sqrt(v1.x * v1.x + v1.z * v1.z);
   else
      *v2 = Vector3<T>(0, v1.z,-v1.y) /
            std::sqrt(v1.y * v1.y + v1.z * v1.z);
   *v3 = Cross(v1, *v2);
#endif
}

// Vector4 definition
template<typename T>
CPU_GPU Vector4<T>::Vector4() : x(0),y(0),z(0),w(0){}

template<typename T>
CPU_GPU Vector4<T>::Vector4(T _x, T _y, T _z, T _w) : x(_x),y(_y),z(_z),w(_w){}

template<typename T>
CPU_GPU Vector4<T>::Vector4(const Vector3<T> &v) : x(v.x),y(v.y),z(v.z),w(0) {}

template<typename T>
CPU_GPU Vector4<T>::Vector4(const Vector3<T> &v, T _w) : x(v.x), y(v.y), z(v.z), w(_w) {}

template<typename T>
CPU_GPU Vector4<T>::operator Vector3<T>() const {
    return Vector3<T>(x, y, z);
}

// Vector2 definition
template<typename T>
CPU_GPU Vector2<T>::Vector2() : x(0), y(0) {}

template<typename T>
CPU_GPU Vector2<T>::Vector2(T _x, T _y) : x(_x), y(_y) {}

#endif