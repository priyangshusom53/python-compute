#ifndef BOUNDS_H
#define BOUNDS_H

#include "cudadefines.h"
#include "point.h"
#include "vector.h"
#include "mathconstants.h"


struct Bounds3f{
   Point3f pMin;
   Point3f pMax;
   
   CPU_GPU Bounds3f();
   CPU_GPU Bounds3f(const Point3f &p); 
   CPU_GPU Bounds3f(const Point3f &p1, const Point3f &p2);
   CPU_GPU Point3f &operator[](int i) const;
   CPU_GPU Point3f &operator[](int i);
   CPU_GPU Vector3f Diagonal() const;
   CPU_GPU float  SurfaceArea() const;
   CPU_GPU float Volume() const;
   CPU_GPU int  MaximumExtent() const;
   CPU_GPU Point3f Lerp(Point3f &t) const;
   CPU_GPU Vector3f Offset(const Point3f &p) const;
   CPU_GPU static Bounds3f Union(const Bounds3f &b1, const Bounds3f &b2);
   CPU_GPU static Bounds3f Intersection(const Bounds3f &b1, const Bounds3f &b2);
   CPU_GPU static bool Overlaps(const Bounds3f &b1, const Bounds3f &b2);
   CPU_GPU static bool Inside(const Point3f &p, const Bounds3f &b);
   CPU_GPU static Bounds3f Expand(const Bounds3f &b, float delta);
};

#endif