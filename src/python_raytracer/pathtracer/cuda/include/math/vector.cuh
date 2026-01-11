#ifndef MATH_VECTOR_CUH
#define MATH_VECTOR_CUH

// use float4 everywhere

__device__ __inline__ float value_at(const float4 &v, unsigned int i)
{
   if (i == 0) return v.x;
   if (i == 1) return v.y;
   if (i == 2) return v.z;
   if (i == 3) return v.w;
   return 0;
}

__device__ __forceinline__ float3 to_float3(const float4 v){
   return make_float3(v.x,v.y,v.z);
}

__device__ __forceinline__ float4 make_vec()
{
   return make_float4(0, 0, 0, 0);
}

__device__ __forceinline__ float4 make_vec(float x, float y, float z)
{
   return make_float4(x, y, z, 0);
}

__device__ __forceinline__ float4 make_vec(float3 v){
   return make_float4(v.x,v.y,v.z,0);
}

__device__ __forceinline__ float4 make_vec(float4 v)
{
   return make_float4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ float4 make_point()
{
   return make_float4(0, 0, 0, 1);
}

__device__ __forceinline__ float4 make_point(float x, float y, float z)
{
   return make_float4(x, y, z, 1);
}

__device__ __forceinline__ float4 make_point(float3 v){
   return make_float4(v.x,v.y,v.z,1);
}

__device__ __forceinline__ float4 make_point(float4 v)
{
   return make_float4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ float4 make_normal(float x, float y, float z)
{
   return make_float4(x, y, z, 0);
}

__device__ __forceinline__ float4 make_normal(float4 v)
{
   return make_float4(v.x, v.y, v.z, 0);
}

__device__ __forceinline__ float4 operator+=(float4 &a, const float4 b)
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

__device__ __forceinline__ float3 operator+(const float3 a, const float3 b){
   float3 c;
   c.x = a.x+b.x;
   c.y = a.y+b.y;
   c.z = a.z+b.z;
   return c;
}

__device__ __forceinline__ float2 operator+(const float2 a, const float2 b){
   float2 c;
   c.x = a.x+b.x;
   c.y = a.y+b.y;
   return c;
}

__device__ __forceinline__ float4 operator-=(float4 &a, const float4 b)
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

__device__ __forceinline__ float3 operator-(const float3 a, const float3 b){
   float3 c;
   c.x = a.x-b.x;
   c.y = a.y-b.y;
   c.z = a.z-b.z;
   return c;
}

__device__ __forceinline__ float2 operator-(const float2 a, const float2 b){
   float2 c;
   c.x = a.x-b.x;
   c.y = a.y-b.y;
   return c;
}

__device__ __forceinline__ float4 operator*=(float4 &a, const float s)
{
   a.x *= s;
   a.y *= s;
   a.z *= s;
   a.w *= s;
   return a;
}

__device__ __forceinline__ float4 operator*(const float4 a, const float s)
{
   float4 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float4 operator*(const float s, const float4 a)
{
   float4 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float3 operator*=(float3 &a, const float s)
{
   a.x *= s;
   a.y *= s;
   a.z *= s;
   return a;
}

__device__ __forceinline__ float3 operator*(const float3 a, const float s)
{
   float3 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float3 operator*(const float s, const float3 a)
{
   float3 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float2 operator*=(float2 &a, const float s)
{
   a.x *= s;
   a.y *= s;
   return a;
}

__device__ __forceinline__ float2 operator*(const float2 a, const float s)
{
   float2 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float2 operator*(const float s, const float2 a)
{
   float2 b = a;
   b *= s;
   return b;
}

__device__ __forceinline__ float4 operator/(const float4 v, const float s){
   return make_float4(v.x/s,v.y/s,v.z/s,v.w);
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

__device__ __forceinline__ float lengthSquared(const float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

__device__ __forceinline__ float length(const float4 v) { return sqrtf(lengthSquared(v)); }

__device__ __forceinline__ float length(const float3 v) { return sqrtf(lengthSquared(v)); }

__device__ __forceinline__ float4 normalize(const float4 v)
{
   float inv_length = 1 / length(v);
   float4 _v = v * inv_length;
   _v.w = v.w;
   return _v;
}

__device__ __forceinline__ float3 normalize(const float3 v)
{
   float inv_length = 1 / length(v);
   float3 _v = v * inv_length;
   return _v;
}

__device__ __forceinline__ float minComponent(const float4 v) { return fminf(v.x, fminf(v.y, v.z)); }

__device__ __forceinline__ float maxComponent(const float4 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }

__device__ __forceinline__ int maxDim(const float4 v)
{
   return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

__device__ __forceinline__ float4 vecAbs(const float4 v)
{
   return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

__device__ __forceinline__ float4 vecMin(const float4 v1, const float4 v2)
{
   return make_float4(fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z), fminf(v1.w, v2.w));
}

__device__ __forceinline__ float4 vecMax(const float4 v1, const float4 v2)
{
   return make_float4(fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z), fmaxf(v1.w, v2.w));
}

__device__ __forceinline__ float4 permute3(const float4 v,const unsigned int x, const unsigned int y, const unsigned int z){
   return make_float4(value_at(v,x), value_at(v,y), value_at(v,z), v.w);
}

#endif