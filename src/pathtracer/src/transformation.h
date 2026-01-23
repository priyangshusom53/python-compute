#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "cudadefines.h"
#include "matrix.h"
#include "vector.h"
#include "point.h"
#include "normal.h"

struct Transform {
	Matrix4f matrix, invMatrix;
	CPU_GPU Transform();
	CPU_GPU Transform(const Matrix4f& matrix);
	CPU_GPU Transform(const Matrix4f& matrix, const Matrix4f& invMatrix);
	CPU_GPU static Transform Identity();
	CPU_GPU Point3f TransformPoint(const Point3f& p);
	CPU_GPU Point4f TransformPoint(const Point4f& p);
	CPU_GPU Vector4f TransformVector(const Vector4f& v);
	CPU_GPU Normal3f TransformNormal(const Normal3f& n);
};

#endif