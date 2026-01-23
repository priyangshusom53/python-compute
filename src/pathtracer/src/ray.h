#ifndef RAY_H
#define RAY_H

#include "cudadefines.h"
#include "point.h"
#include "vector.h"
#include "mathconstants.h"

struct Ray{
   Point3f o;
   Vector3f d;
   float tMax;
   float time;

   CPU_GPU Ray();
   CPU_GPU Ray(const Point3f &o, const Vector3f &d, float tMax, float time);
   CPU_GPU Point3f operator()(float t) const;
};

CPU_GPU Ray::Ray() : tMax(infinity_f()), time(0.f) {}
CPU_GPU Ray::Ray(const Point3f &o, const Vector3f &d, float tMax = infinity_f(), float time = 0.f) : o(o), d(d), tMax(tMax), time(time) {}
CPU_GPU Point3f Ray::operator()(float t) const {
   return o + d * t;
}

#endif