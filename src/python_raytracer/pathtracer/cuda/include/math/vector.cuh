#ifndef MATH_VECTOR_CUH
#define MATH_VECTOR_CUH

// use float4 everywhere

__device__ __forceinline__ float4 &operator+=(float4 &a, const float4 b)
{
   a.x += b.x;
   a.y += b.y;
   a.z += b.z;
   a.w += b.w;
   return a;
}

__device__ __forceinline__
    float4
    operator+(const float4 a, const float4 b)
{
   float4 c = a;
   c += b;
   return c;
}

__device__ __forceinline__ float4 &operator-=(float4 &a, const float4 b)
{
   a.x -= b.x;
   a.y -= b.y;
   a.z -= b.z;
   a.w -= b.w;
   return a;
}

__device__ __forceinline__
    float4
    operator-(const float4 a, const float4 b)
{
   float4 c = a;
   c -= b;
   return c;
}

__device__ __forceinline__ float4 &operator*=(float4 &a, const float s)
{
   a.x *= s;
   a.y *= s;
   a.z *= s;
   a.w *= s;
   return a;
}

__device__ __forceinline__
    float4
    operator*(const float4 a, const float s)
{
   float4 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__
    float4
    operator*(const float s, const float4 a)
{
   float4 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float dot3(const float4 &v1, const float4 &v2)
{
   return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ float dot4(const float4 &v1, const float4 &v2)
{
   return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

__device__ __forceinline__ float absDot3(const float4 v1, const float4 v2)
{
   return fabsf(dot3(v1, v2));
}

__device__ __forceinline__ float4 cross3(const float4 v1, const float4 v2)
{
   float v1x = v1.x, v1y = v1.y, v1z = v1.z;
   float v2x = v2.x, v2y = v2.y, v2z = v2.z;
   return make_float4((v1y * v2z) - (v1z * v2y),
                      (v1z * v2x) - (v1x * v2z),
                      (v1x * v2y) - (v1y * v2x), 1);
}

__device__ __forceinline__ float lengthSquared(const float4 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

__device__ __forceinline__ float length(const float4 v) { return sqrtf(lengthSquared(v)); }

__device__ __forceinline__ float4 normalize(const float4 v)
{
   float inv_length = 1 / length(v);
   float4 _v = v * inv_length;
   _v.w = 1;
   return _v;
}

__device__ __forceinline__ float minComponent(const float4 v) { return fminf(v.x, fminf(v.y, v.z)); }

__device__ __forceinline__ float maxComponent(const float4 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }

__device__ __forceinline__ int maxDim(const float4 v)
{
   return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

__device__ __forceinline__ float4 vecMin(const float4 v1, const float4 v2)
{
   return make_float4(fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z), fminf(v1.w, v2.w));
}

__device__ __forceinline__ float4 vecMax(const float4 v1, const float4 v2)
{
   return make_float4(fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z), fmaxf(v1.w, v2.w));
}

#endif