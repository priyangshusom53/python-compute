#ifndef BOUNDS_H
#define BOUNDS_H

#include "cudadefines.h"
#include "point.h"
#include "vector.h"
#include "mathconstants.h"
#include "utils.h"



struct Bounds3f{
   Point3f pMin;
   Point3f pMax;
   
   CPU_GPU Bounds3f();
   CPU_GPU Bounds3f(const Point3f &p); 
   CPU_GPU Bounds3f(const Point3f &p1, const Point3f &p2);
   CPU_GPU const Point3f &operator[](int idx) const;
   CPU_GPU Point3f &operator[](int idx);
   CPU_GPU Vector3f Diagonal() const;
   CPU_GPU float  SurfaceArea() const;
   CPU_GPU float Volume() const;
   CPU_GPU int  MaximumExtent() const;
   CPU_GPU Point3f Lerp(const Point3f &t) const;
   CPU_GPU Vector3f Offset(const Point3f &p) const;
   CPU_GPU static Bounds3f Union(const Bounds3f& b, const Point3f& p);
   CPU_GPU static Bounds3f Union(const Bounds3f &b1, const Bounds3f &b2);
   CPU_GPU static Bounds3f Intersection(const Bounds3f &b1, const Bounds3f &b2);
   CPU_GPU static bool Overlaps(const Bounds3f &b1, const Bounds3f &b2);
   CPU_GPU static bool Inside(const Point3f &p, const Bounds3f &b);
   CPU_GPU static Bounds3f Expand(const Bounds3f &b, float delta);
};

struct Bounds2f{
	Point2f pMin;
	Point2f pMax;
	CPU_GPU Bounds2f();
	CPU_GPU Bounds2f(const Point2f& p);
	CPU_GPU Bounds2f(const Point2f& p1, const Point2f& p2);
};


// Bounds3 definition
CPU_GPU Bounds3f::Bounds3f() : pMin(Point3f(max_f())), pMax(Point3f(lowest_f())) {}

CPU_GPU Bounds3f::Bounds3f(const Point3f& p) : pMin(p), pMax(p) {}

CPU_GPU Bounds3f::Bounds3f(const Point3f& p1, const Point3f& p2){
   pMin = Point3f::Min(p1, p2);
   pMax = Point3f::Max(p1, p2);
}

CPU_GPU const Point3f& Bounds3f::operator[](int idx) const {
   return (&pMin)[idx];
}

CPU_GPU Point3f& Bounds3f::operator[](int idx) {
	return (&pMin)[idx];
}

CPU_GPU Vector3f Bounds3f::Diagonal() const {
	return pMax - pMin;
}

CPU_GPU float Bounds3f::SurfaceArea() const {
	Vector3f d = Diagonal();
	return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
}

CPU_GPU float Bounds3f::Volume() const {
	Vector3f d = Diagonal();
	return d.x * d.y * d.z;
}

/*	
*	Returns the dimension in which Diagonal Vector has max component
*/
CPU_GPU int  Bounds3f::MaximumExtent() const {
	return Vector3f::MaxDimension(Diagonal());
}

CPU_GPU Point3f Bounds3f::Lerp(const Point3f& t) const {
	return Point3f(	::Lerp(t.x, pMin.x, pMax.x),
					::Lerp(t.y, pMin.y, pMax.y),
					::Lerp(t.z, pMin.z, pMax.z));
}

CPU_GPU Vector3f Bounds3f::Offset(const Point3f& p) const {
	Vector3f o = p - pMin;
	if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
	if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
	if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
	return o;
}

CPU_GPU Bounds3f Bounds3f::Union(const Bounds3f& b, const Point3f& p) {

#ifdef __CUDA_ARCH__
	Bounds3f(Point3f(min(p.x, b.pMin.x),
					min(p.y, b.pMin.y),
					min(p.z, b.pMin.z)),
			Point3f(max(p.x, b.pMax.x),
					max(p.y, b.pMax.y),
					max(p.z, b.pMax.z)));
#else
	return Bounds3f(Point3f(std::min(p.x, b.pMin.x),
							std::min(p.y, b.pMin.y),
							std::min(p.z, b.pMin.z)),
					Point3f(std::max(p.x, b.pMax.x),
							std::max(p.y, b.pMax.y),
							std::max(p.z, b.pMax.z)));
#endif
}

CPU_GPU Bounds3f Bounds3f::Union(const Bounds3f& b1, const Bounds3f& b2) {

#ifdef __CUDA_ARCH__
	Bounds3f(Point3f(min(b1.pMin.x, b2.pMin.x),
					 min(b1.pMin.y, b2.pMin.y),
					 min(b1.pMin.z, b2.pMin.z)),
			 Point3f(max(b1.pMax.x, b2.pMax.x),
					 max(b1.pMax.y, b2.pMax.y),
					 max(b1.pMax.z, b2.pMax.z)));
#else
	return Bounds3f(Point3f(std::min(b1.pMin.x, b2.pMin.x),
							std::min(b1.pMin.y, b2.pMin.y),
							std::min(b1.pMin.z, b2.pMin.z)),
					Point3f(std::max(b1.pMax.x, b2.pMax.x),
							std::max(b1.pMax.y, b2.pMax.y),
							std::max(b1.pMax.z, b2.pMax.z)));
#endif
}

CPU_GPU Bounds3f Bounds3f::Intersection(const Bounds3f& b1, const Bounds3f& b2) {

#ifdef __CUDA_ARCH__
	Bounds3f(Point3f(max(b1.pMin.x, b2.pMin.x),
					 max(b1.pMin.y, b2.pMin.y),
					 max(b1.pMin.z, b2.pMin.z)),
			 Point3f(min(b1.pMax.x, b2.pMax.x),
					 min(b1.pMax.y, b2.pMax.y),
					 min(b1.pMax.z, b2.pMax.z)));
#else
	return Bounds3f(Point3f(std::max(b1.pMin.x, b2.pMin.x),
							std::max(b1.pMin.y, b2.pMin.y),
							std::max(b1.pMin.z, b2.pMin.z)),
					Point3f(std::min(b1.pMax.x, b2.pMax.x),
							std::min(b1.pMax.y, b2.pMax.y),
							std::min(b1.pMax.z, b2.pMax.z)));
#endif
}

CPU_GPU bool Bounds3f::Overlaps(const Bounds3f& b1, const Bounds3f& b2) {
	bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
	bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
	bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
	return (x && y && z);
}

CPU_GPU bool Bounds3f::Inside(const Point3f& p, const Bounds3f& b) {
	return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
			p.y >= b.pMin.y && p.y <= b.pMax.y &&
			p.z >= b.pMin.z && p.z <= b.pMax.z);
}

CPU_GPU Bounds3f Bounds3f::Expand(const Bounds3f& b, float delta) {
	return Bounds3f(b.pMin - Vector3f(delta),
					b.pMax + Vector3f(delta));
}

//	Bounds2 definition
CPU_GPU Bounds2f::Bounds2f() : pMin(max_f()), pMax(lowest_f()) {}

CPU_GPU Bounds2f::Bounds2f(const Point2f& p) : pMin(p), pMax(p) {}

CPU_GPU Bounds2f::Bounds2f(const Point2f& p1, const Point2f& p2) : pMin(p1), pMax(p2) {}

#endif