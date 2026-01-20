#include "vector.h"

template<typename T>
CPU_GPU Vector3<T>::Vector3() : x(0),y(0),z(0){}

template<typename T>
CPU_GPU Vector3<T>::Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

template<typename T>
CPU_GPU T Vector3<T>::operator[](const int idx){
   T arr[3] = {x,y,z};
   return arr[idx % 3];
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
CPU_GPU Vector4<T>::Vector4() : x(0),y(0),z(0),w(0){}

template<typename T>
CPU_GPU Vector4<T>::Vector4(T _x, T _y, T _z, T _w) : x(_x),y(_y),z(_z),w(_w){}

template<typename T>
CPU_GPU Vector4<T>::Vector4(const Vector3<T>&){}

// template<typename T>
// CPU_GPU Vector4<T>& Vector4<T>::operator+=(const Vector4<T>& other){
//    this->x+=other.x;
//    this->y+=other.y;
//    this->z+=other.z;
//    return *this;
// }

// template<typename T>
// CPU_GPU Vector4<T> Vector4<T>::operator+(const Vector4<T>& other) const{
//    Vector4<T> result = *this;
//    return result+=other;
// }