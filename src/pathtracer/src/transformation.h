#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "cudadefines.h"
#include "matrix.h"
#include "vector.h"
#include "point.h"
#include "normal.h"
#include "bounds.h"
#include "ray.h"

struct CUDA_ALIGN(16) Transform {
	Matrix4f matrix, invMatrix;
	CPU_GPU Transform();
	CPU_GPU Transform(const Matrix4f& _matrix);
	CPU_GPU Transform(const Matrix4f& _matrix, const Matrix4f& _invMatrix);
	CPU_GPU static Transform Identity();
	CPU_GPU static Transform Translate(float x, float y, float z);
	CPU_GPU static Transform Translate(const Vector3f& v);
	CPU_GPU static Transform Scale(float x, float y, float z);
	CPU_GPU static Transform Scale(const Vector3f& v);
	CPU_GPU static Transform Rotation(const Matrix4f& m);
	CPU_GPU Point4f TransformPoint(const Point3f& p) const;
	CPU_GPU Point4f TransformPoint(const Point4f& p) const;
	CPU_GPU Vector3f TransformVector(const Vector3f& v) const;
	CPU_GPU Normal3f TransformNormal(const Normal3f& n) const;
	CPU_GPU Bounds3f TransformBounds(const Bounds3f& b) const;
	CPU_GPU Ray TransformRay(const Ray& r) const;
	CPU_GPU Transform InverseTransform() const;
};


// Transform definition
CPU_GPU INLINE Transform::Transform(){
	matrix = Matrix4f::Identity();
	invMatrix = Matrix4f::Identity();
}

CPU_GPU INLINE Transform::Transform(const Matrix4f& _matrix) {
	matrix = _matrix;
	invMatrix = _matrix.Inverse();
}

CPU_GPU INLINE Transform::Transform(const Matrix4f& _matrix, const Matrix4f& _invMatrix) {

}

CPU_GPU INLINE Transform Transform::Identity() {
	Transform t;
	t.matrix = Matrix4f::Identity();
	t.invMatrix = Matrix4f::Identity();
	return t;
}

CPU_GPU INLINE Transform Transform::Translate(float x, float y, float z) {
	Transform t = Transform::Identity();
	t.matrix.elements[0][3] = x;
	t.matrix.elements[1][3] = y;
	t.matrix.elements[2][3] = z;

	t.invMatrix.elements[0][3] = -x;
	t.invMatrix.elements[1][3] = -y;
	t.invMatrix.elements[2][3] = -z;

	return t;
}

CPU_GPU INLINE Transform Transform::Translate(const Vector3f& v) {
	Transform t = Transform::Identity();
	t.matrix.elements[0][3] = v.x;
	t.matrix.elements[1][3] = v.y;
	t.matrix.elements[2][3] = v.z;

	t.invMatrix.elements[0][3] = -v.x;
	t.invMatrix.elements[1][3] = -v.y;
	t.invMatrix.elements[2][3] = -v.z;

	return t;
}

CPU_GPU INLINE Transform Transform::Scale(float x, float y, float z) {

	Transform t = Transform::Identity();
	t.matrix.elements[0][0] = x;
	t.matrix.elements[1][1] = y;
	t.matrix.elements[2][2] = z;

	t.invMatrix.elements[0][0] = 1 / x;
	t.invMatrix.elements[1][1] = 1 / y;
	t.invMatrix.elements[2][2] = 1 / z;
	return t;
}

CPU_GPU INLINE Transform Transform::Scale(const Vector3f& v) {
	Transform t = Transform::Identity();
	t.matrix.elements[0][0] = v.x;
	t.matrix.elements[1][1] = v.y;
	t.matrix.elements[2][2] = v.z;

	t.invMatrix.elements[0][0] = 1 / v.x;
	t.invMatrix.elements[1][1] = 1 / v.y;
	t.invMatrix.elements[2][2] = 1 / v.z;
	return t;
}

CPU_GPU INLINE Transform Transform::Rotation(const Matrix4f& m) {
	Transform t;
	t.matrix = m;
	t.invMatrix = m.Inverse();
	return t;
}

CPU_GPU INLINE Point4f Transform::TransformPoint(const Point3f& p) const {
	Point4f _p;
	_p.x = matrix.elements[0][0] * p.x + matrix.elements[0][1] * p.y +
		matrix.elements[0][2] * p.z + matrix.elements[0][3] * 1.f;

	_p.y = matrix.elements[1][0] * p.x + matrix.elements[1][1] * p.y +
		matrix.elements[1][2] * p.z + matrix.elements[1][3] * 1.f;

	_p.z = matrix.elements[2][0] * p.x + matrix.elements[2][1] * p.y +
		matrix.elements[2][2] * p.z + matrix.elements[2][3] * 1.f;

	_p.w = matrix.elements[3][0] * p.x + matrix.elements[3][1] * p.y +
		matrix.elements[3][2] * p.z + matrix.elements[3][3] * 1.f;
	return _p;
}

CPU_GPU INLINE Point4f Transform::TransformPoint(const Point4f& p) const {
	Point4f _p;
	_p.x = matrix.elements[0][0] * p.x + matrix.elements[0][1] * p.y +
		matrix.elements[0][2] * p.z + matrix.elements[0][3] * p.w;

	_p.y = matrix.elements[1][0] * p.x + matrix.elements[1][1] * p.y +
		matrix.elements[1][2] * p.z + matrix.elements[1][3] * p.w;

	_p.z = matrix.elements[2][0] * p.x + matrix.elements[2][1] * p.y +
		matrix.elements[2][2] * p.z + matrix.elements[2][3] * p.w;

	_p.w = matrix.elements[3][0] * p.x + matrix.elements[3][1] * p.y +
		matrix.elements[3][2] * p.z + matrix.elements[3][3] * p.w;
	
	return _p;
}

CPU_GPU INLINE Vector3f Transform::TransformVector(const Vector3f& v) const {
	Vector3f _v;
	_v.x = matrix.elements[0][0] * v.x + matrix.elements[0][1] * v.y +
		matrix.elements[0][2] * v.z;

	_v.y = matrix.elements[1][0] * v.x + matrix.elements[1][1] * v.y +
		matrix.elements[1][2] * v.z;

	_v.z = matrix.elements[2][0] * v.x + matrix.elements[2][1] * v.y +
		matrix.elements[2][2] * v.z;

	return _v;
}

CPU_GPU INLINE Normal3f Transform::TransformNormal(const Normal3f& n) const {
	const Matrix4f s = invMatrix.Transpose();
	Normal3f _n;
	_n.x = matrix.elements[0][0] * n.x + matrix.elements[0][1] * n.y +
		matrix.elements[0][2] * n.z;

	_n.y = matrix.elements[1][0] * n.x + matrix.elements[1][1] * n.y +
		matrix.elements[1][2] * n.z;

	_n.z = matrix.elements[2][0] * n.x + matrix.elements[2][1] * n.y +
		matrix.elements[2][2] * n.z;

	return _n;
}

CPU_GPU INLINE Bounds3f Transform::TransformBounds(const Bounds3f& b) const {
	const Transform& t = *this;
	Bounds3f ret(Point3f(t.TransformPoint(Point3f(b.pMin.x, b.pMin.y, b.pMin.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMax.x, b.pMin.y, b.pMin.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMin.x, b.pMax.y, b.pMin.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMin.x, b.pMin.y, b.pMax.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMin.x, b.pMax.y, b.pMax.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMax.x, b.pMax.y, b.pMin.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMax.x, b.pMin.y, b.pMax.z))));
	ret = Bounds3f::Union(ret, Point3f(t.TransformPoint(Point3f(b.pMax.x, b.pMax.y, b.pMax.z))));
	return ret;
}

CPU_GPU Ray Transform::TransformRay(const Ray& r)const {
	const Transform& t = *this;
	Ray _r;
	_r.o = Point3f(t.TransformPoint(r.o));
	_r.d = t.TransformVector(r.d);
	_r.tMax = r.tMax;
	_r.time = r.time;
	return _r;
}

CPU_GPU Transform Transform::InverseTransform() const {
	Transform _t;
	_t.matrix = invMatrix;
	_t.invMatrix = matrix;
	return _t;
}

#endif