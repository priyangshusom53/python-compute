#ifndef MATH_TRANSFORM_CUH
#define MATH_TRANSFORM_CUH

#include "math/vector.cuh"
#include "ray.cuh"

struct Ray;

// matrix is row major
struct __align__(16) Mat4
{
   float4 r0;
   float4 r1;
   float4 r2;
   float4 r3;
};

__device__ __forceinline__ Mat4 identity()
{
   Mat4 m;
   m.r0 = make_float4(1.f, 0.f, 0.f, 0.f);
   m.r1 = make_float4(0.f, 1.f, 0.f, 0.f);
   m.r2 = make_float4(0.f, 0.f, 1.f, 0.f);
   m.r3 = make_float4(0.f, 0.f, 0.f, 1.f);
   return m;
}

__device__ __forceinline__ Mat4 &mat4_transpose(const Mat4 &m)
{
   Mat4 _m;
   _m.r0 = make_float4(m.r0.x, m.r1.x, m.r2.x, m.r3.x);
   _m.r1 = make_float4(m.r0.y, m.r1.y, m.r2.y, m.r3.y);
   _m.r2 = make_float4(m.r0.z, m.r1.z, m.r2.z, m.r3.z);
   _m.r3 = make_float4(m.r0.w, m.r1.w, m.r2.w, m.r3.w);
   return _m;
}

__device__ __forceinline__ Mat4 mat4_inverse(const Mat4 &m)
{
   float a00 = m.r0.x, a01 = m.r0.y, a02 = m.r0.z, a03 = m.r0.w;
   float a10 = m.r1.x, a11 = m.r1.y, a12 = m.r1.z, a13 = m.r1.w;
   float a20 = m.r2.x, a21 = m.r2.y, a22 = m.r2.z, a23 = m.r2.w;
   float a30 = m.r3.x, a31 = m.r3.y, a32 = m.r3.z, a33 = m.r3.w;

   float b00 = a00 * a11 - a01 * a10;
   float b01 = a00 * a12 - a02 * a10;
   float b02 = a00 * a13 - a03 * a10;
   float b03 = a01 * a12 - a02 * a11;
   float b04 = a01 * a13 - a03 * a11;
   float b05 = a02 * a13 - a03 * a12;
   float b06 = a20 * a31 - a21 * a30;
   float b07 = a20 * a32 - a22 * a30;
   float b08 = a20 * a33 - a23 * a30;
   float b09 = a21 * a32 - a22 * a31;
   float b10 = a21 * a33 - a23 * a31;
   float b11 = a22 * a33 - a23 * a32;

   float det =
       b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

   float invDet = 1.0f / det;

   Mat4 inv;

   inv.r0.x = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
   inv.r0.y = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
   inv.r0.z = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
   inv.r0.w = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;

   inv.r1.x = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
   inv.r1.y = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
   inv.r1.z = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
   inv.r1.w = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;

   inv.r2.x = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
   inv.r2.y = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
   inv.r2.z = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
   inv.r2.w = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;

   inv.r3.x = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
   inv.r3.y = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
   inv.r3.z = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
   inv.r3.w = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;

   return inv;
}

__device__ __forceinline__ Mat4 mat4_inverse_affine(const Mat4 &m)
{
   Mat4 inv;

   // Inverse of upper-left 3x3
   float det =
       m.r0.x * (m.r1.y * m.r2.z - m.r1.z * m.r2.y) -
       m.r0.y * (m.r1.x * m.r2.z - m.r1.z * m.r2.x) +
       m.r0.z * (m.r1.x * m.r2.y - m.r1.y * m.r2.x);

   float invDet = 1.0f / det;

   inv.r0.x = (m.r1.y * m.r2.z - m.r1.z * m.r2.y) * invDet;
   inv.r0.y = -(m.r0.y * m.r2.z - m.r0.z * m.r2.y) * invDet;
   inv.r0.z = (m.r0.y * m.r1.z - m.r0.z * m.r1.y) * invDet;

   inv.r1.x = -(m.r1.x * m.r2.z - m.r1.z * m.r2.x) * invDet;
   inv.r1.y = (m.r0.x * m.r2.z - m.r0.z * m.r2.x) * invDet;
   inv.r1.z = -(m.r0.x * m.r1.z - m.r0.z * m.r1.x) * invDet;

   inv.r2.x = (m.r1.x * m.r2.y - m.r1.y * m.r2.x) * invDet;
   inv.r2.y = -(m.r0.x * m.r2.y - m.r0.y * m.r2.x) * invDet;
   inv.r2.z = (m.r0.x * m.r1.y - m.r0.y * m.r1.x) * invDet;

   // Inverse translation
   float tx = m.r0.w;
   float ty = m.r1.w;
   float tz = m.r2.w;

   inv.r0.w = -(inv.r0.x * tx + inv.r0.y * ty + inv.r0.z * tz);
   inv.r1.w = -(inv.r1.x * tx + inv.r1.y * ty + inv.r1.z * tz);
   inv.r2.w = -(inv.r2.x * tx + inv.r2.y * ty + inv.r2.z * tz);

   // Last row
   inv.r3 = make_float4(0.f, 0.f, 0.f, 1.f);

   return inv;
}

__device__ __forceinline__ float4 mat_mul(const Mat4 &m, const float4 v)
{
   float4 _v;
   _v.x = dot4(m.r0, v);
   _v.y = dot4(m.r1, v);
   _v.z = dot4(m.r2, v);
   _v.w = dot4(m.r3, v);
   return _v;
}

__device__ __forceinline__ Mat4 &mat_mul(const Mat4 &m1, const Mat4 &m2)
{
   Mat4 _m;
   Mat4 m2T = mat4_transpose(m2);
   _m.r0 = mat_mul(m1, m2T.r0);
   _m.r1 = mat_mul(m1, m2T.r0);
   _m.r2 = mat_mul(m1, m2T.r0);
   _m.r3 = mat_mul(m1, m2T.r0);
   return mat4_transpose(_m);
}

__device__ __forceinline__ Mat4 translate(const float4 t)
{
   Mat4 m = identity();
   m.r0.w = t.x;
   m.r1.w = t.y;
   m.r2.w = t.z;
   return m;
}

__device__ __forceinline__ Mat4 scale(const float4 s)
{
   Mat4 m;
   m.r0 = make_float4(s.x, 0.f, 0.f, 0.f);
   m.r1 = make_float4(0.f, s.y, 0.f, 0.f);
   m.r2 = make_float4(0.f, 0.f, s.z, 0.f);
   return m;
}

__device__ __forceinline__ float4 transform_vector(const Mat4 &m, const float4 v)
{
   return make_float4(
       dot3(m.r0, v),
       dot3(m.r1, v),
       dot3(m.r2, v),
       v.w);
}

__device__ __forceinline__ float4 transform_point(const Mat4 &m, const float4 p)
{
   return make_float4(
       dot4(m.r0, p),
       dot4(m.r1, p),
       dot4(m.r2, p),
       dot4(m.r3, p));
}

// __device__ __forceinline__ float4 transform_normal(const Mat4 &m, const float4 n) {}

struct __align__(16) Transform
{
   Mat4 matrix;
   Mat4 inv_matrix;
};

__device__ __forceinline__ Transform &identity_transform()
{
   Transform _t;
   _t.matrix = identity();
   _t.inv_matrix = identity();
   return _t;
}

__device__ __forceinline__ Transform &translate_transform(float4 t)
{
   Transform _t;
   _t.matrix = translate(t);
   _t.inv_matrix = translate(-1 * t);
   return _t;
}

__device__ __forceinline__ Transform &scale_transform(const float4 s)
{
   Transform _t;
   _t.matrix = scale(s);
   _t.inv_matrix = scale(make_float4(1 / s.x, 1 / s.y, 1 / s.z, 0));
   return _t;
}

__device__ __forceinline__ float4 transform_vector(const Transform &t, const float4 v)
{
   return transform_vector(t.matrix, v);
}

__device__ __forceinline__ float4 transform_point(const Transform &t, const float4 p)
{
   return transform_point(t.matrix, p);
}

__device__ __forceinline__ float4 transform_normal(const Transform &t, const float4 n)
{
   Mat4 _s = mat4_transpose(t.inv_matrix);
   float4 _n = mat_mul(_s, n);
   return _n;
}

__device__ __forceinline__ Ray &transform_ray(const Transform &t, const Ray &ray)
{
   Ray _ray;
   _ray.o = transform_point(t.matrix, ray.o);
   _ray.d = transform_vector(t.matrix, ray.d);
   return _ray;
}

#endif