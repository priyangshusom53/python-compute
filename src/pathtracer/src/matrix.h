#ifndef MATRIX_H
#define MATRIX_H

#include "cudadefines.h"
#include "vector.h"
#include "utils.h"

#include<iostream>
#include <iomanip>
#include <cmath>

template<typename T>
struct CUDA_ALIGN(16) Matrix4{
	T elements[4][4];
	CPU_GPU Matrix4();
	CPU_GPU Matrix4(const T _elements[16]);
	/*template<int N> CPU_GPU Matrix4(const T(&_r0)[N], const T(&_r1)[N], const T(&_r2)[N], const T(&_r3)[N]);*/
	CPU_GPU static Matrix4<T> Identity();
	CPU_GPU T Determinant() const;
	CPU_GPU Matrix4<T> Transpose() const;
	CPU_GPU Matrix4<T> Inverse()  const;
	CPU_GPU Vector4<T> operator*(const Vector4<T>& v) const;
	CPU_GPU Matrix4<T> operator*(const Matrix4<T>& m) const;
	CPU_GPU static Matrix4<T> MatMul(const Matrix4<T>& m1, const Matrix4<T>& m2);
	CPU_GPU static T Determinant(const Matrix4<T>&);
	CPU_GPU static Matrix4<T> Transpose(const Matrix4<T>& m);
	CPU_GPU static Matrix4<T> Inverse(const Matrix4<T>& m);
	CPU_ONLY friend std::ostream& operator<<(std::ostream& out, const Matrix4<T>& m);
};

typedef Matrix4<float> Matrix4f;

//	Matrix4 definition
template<typename T>
CPU_GPU Matrix4<T>::Matrix4(){
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			elements[i][j] = 0;
		}
	}
}

//template<typename T>
//template<int N> 
//CPU_GPU Matrix4<T>::Matrix4(const T (&_r0)[N], 
//	const T (&_r1)[N], const T (&_r2)[N], const T (&_r3)[N]) {
//
//	static_assert(N == 4, "Matrix4 row arrays must have exactly 4 elements");
//	for (int i = 0; i < 4; ++i) {
//		r0[i] = _r0[i];
//		r1[i] = _r1[i];
//		r2[i] = _r2[i];
//		r3[i] = _r3[i];
//	}
//}

template<typename T>
CPU_GPU Matrix4<T>::Matrix4(const T _elements[16]) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			elements[i][j] = _elements[i * 4 + j];
		}
	}
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::Identity() {
	Matrix4<T> m;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m.elements[i][j] = (i == j) ? 1 : 0;
		}
	}
	return m;
}

template<typename T>
CPU_GPU T Matrix4<T>::Determinant() const {
	T c00 = elements[1][1] * (elements[2][2] * elements[3][3] -
		elements[2][3] * elements[3][2]);
	c00 += -elements[1][2] * (elements[2][1] * elements[3][3] -
		elements[2][3] * elements[3][1]);
	c00 += elements[1][3] * (elements[2][1] * elements[3][2] -
		elements[2][2] * elements[3][1]);
	c00 *= elements[0][0];


	T c01 = elements[1][0] * (elements[2][2] * elements[3][3] -
		elements[2][3] * elements[3][2]);
	c01 += -elements[1][2] * (elements[2][0] * elements[3][3] -
		elements[2][3] * elements[3][0]);
	c01 += elements[1][3] * (elements[2][0] * elements[3][2] -
		elements[2][2] * elements[3][0]);
	c01 *= -elements[0][1];


	T c02 = elements[1][0] * (elements[2][1] * elements[3][3] -
		elements[2][3] * elements[3][1]);
	c02 += -elements[1][1] * (elements[2][0] * elements[3][3] -
		elements[2][3] * elements[3][0]);
	c02 += elements[1][3] * (elements[2][0] * elements[3][1] -
		elements[2][1] * elements[3][0]);
	c02 *= elements[0][2];


	T c03 = elements[1][0] * (elements[2][1] * elements[3][2] -
		elements[2][2] * elements[3][1]);
	c03 += -elements[1][1] * (elements[2][0] * elements[3][2] -
		elements[2][2] * elements[3][0]);
	c03 += elements[1][2] * (elements[2][0] * elements[3][1] -
		elements[2][1] * elements[3][0]);
	c03 *= -elements[0][3];
	
	return c00 + c01 + c02 + c03;
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::Transpose() const {

	Matrix4<T> m;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m.elements[j][i] = elements[i][j];
		}
	}
	return m;
}

template<typename T>
CPU_GPU inline T Det3(
	T a00, T a01, T a02,
	T a10, T a11, T a12,
	T a20, T a21, T a22)
{
	return a00 * (a11 * a22 - a12 * a21)
		- a01 * (a10 * a22 - a12 * a20)
		+ a02 * (a10 * a21 - a11 * a20);
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::Inverse()  const {
	Matrix4<T> inv;
	const T det = Determinant();

	// Handle singular matrix
#ifdef __CUDA_ARCH__
	if (det == T(0)) return Matrix4<T>::Identity();
#else
	if (std::abs(det) < T(1e-8)) return Matrix4<T>::Identity();
#endif

	const T invDet = T(1) / det;

	// Cofactors (row-major)
	inv.elements[0][0] = Det3(
		elements[1][1], elements[1][2], elements[1][3],
		elements[2][1], elements[2][2], elements[2][3],
		elements[3][1], elements[3][2], elements[3][3]) * invDet;

	inv.elements[0][1] = -Det3(
		elements[0][1], elements[0][2], elements[0][3],
		elements[2][1], elements[2][2], elements[2][3],
		elements[3][1], elements[3][2], elements[3][3]) * invDet;

	inv.elements[0][2] = Det3(
		elements[0][1], elements[0][2], elements[0][3],
		elements[1][1], elements[1][2], elements[1][3],
		elements[3][1], elements[3][2], elements[3][3]) * invDet;

	inv.elements[0][3] = -Det3(
		elements[0][1], elements[0][2], elements[0][3],
		elements[1][1], elements[1][2], elements[1][3],
		elements[2][1], elements[2][2], elements[2][3]) * invDet;

	inv.elements[1][0] = -Det3(
		elements[1][0], elements[1][2], elements[1][3],
		elements[2][0], elements[2][2], elements[2][3],
		elements[3][0], elements[3][2], elements[3][3]) * invDet;

	inv.elements[1][1] = Det3(
		elements[0][0], elements[0][2], elements[0][3],
		elements[2][0], elements[2][2], elements[2][3],
		elements[3][0], elements[3][2], elements[3][3]) * invDet;

	inv.elements[1][2] = -Det3(
		elements[0][0], elements[0][2], elements[0][3],
		elements[1][0], elements[1][2], elements[1][3],
		elements[3][0], elements[3][2], elements[3][3]) * invDet;

	inv.elements[1][3] = Det3(
		elements[0][0], elements[0][2], elements[0][3],
		elements[1][0], elements[1][2], elements[1][3],
		elements[2][0], elements[2][2], elements[2][3]) * invDet;

	inv.elements[2][0] = Det3(
		elements[1][0], elements[1][1], elements[1][3],
		elements[2][0], elements[2][1], elements[2][3],
		elements[3][0], elements[3][1], elements[3][3]) * invDet;

	inv.elements[2][1] = -Det3(
		elements[0][0], elements[0][1], elements[0][3],
		elements[2][0], elements[2][1], elements[2][3],
		elements[3][0], elements[3][1], elements[3][3]) * invDet;

	inv.elements[2][2] = Det3(
		elements[0][0], elements[0][1], elements[0][3],
		elements[1][0], elements[1][1], elements[1][3],
		elements[3][0], elements[3][1], elements[3][3]) * invDet;

	inv.elements[2][3] = -Det3(
		elements[0][0], elements[0][1], elements[0][3],
		elements[1][0], elements[1][1], elements[1][3],
		elements[2][0], elements[2][1], elements[2][3]) * invDet;

	inv.elements[3][0] = -Det3(
		elements[1][0], elements[1][1], elements[1][2],
		elements[2][0], elements[2][1], elements[2][2],
		elements[3][0], elements[3][1], elements[3][2]) * invDet;

	inv.elements[3][1] = Det3(
		elements[0][0], elements[0][1], elements[0][2],
		elements[2][0], elements[2][1], elements[2][2],
		elements[3][0], elements[3][1], elements[3][2]) * invDet;

	inv.elements[3][2] = -Det3(
		elements[0][0], elements[0][1], elements[0][2],
		elements[1][0], elements[1][1], elements[1][2],
		elements[3][0], elements[3][1], elements[3][2]) * invDet;

	inv.elements[3][3] = Det3(
		elements[0][0], elements[0][1], elements[0][2],
		elements[1][0], elements[1][1], elements[1][2],
		elements[2][0], elements[2][1], elements[2][2]) * invDet;

	return inv;
}

template<typename T>
CPU_GPU Vector4<T> Matrix4<T>::operator*(const Vector4<T>& v) const {
	Vector4<T> _v;
	_v.x = elements[0][0] * v.x + elements[0][1] * v.y +
		elements[0][2] * v.z + elements[0][3] * v.w;
	_v.y = elements[1][0] * v.x + elements[1][1] * v.y +
		elements[1][2] * v.z + elements[1][3] * v.w;
	_v.z = elements[2][0] * v.x + elements[2][1] * v.y +
		elements[2][2] * v.z + elements[2][3] * v.w;
	_v.w = elements[3][0] * v.x + elements[3][1] * v.y +
		elements[3][2] * v.z + elements[3][3] * v.w;
	return _v;
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::operator*(const Matrix4<T>& m) const {
	Matrix4<T> _m;

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			_m.elements[i][j] =
				elements[i][0] * m.elements[0][j] +
				elements[i][1] * m.elements[1][j] +
				elements[i][2] * m.elements[2][j] +
				elements[i][3] * m.elements[3][j];
		}
	}

	return _m;
	
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::MatMul(const Matrix4<T>& m1, const Matrix4<T>& m2) {
	return m1 * m2;
}

template<typename T>
CPU_GPU T Matrix4<T>::Determinant(const Matrix4<T>& m) {
	return m.Determinant();
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::Transpose(const Matrix4<T>& m) {
	return m.Transpose();
}

template<typename T>
CPU_GPU Matrix4<T> Matrix4<T>::Inverse(const Matrix4<T>& m) {
	return m.Inverse();
}

template<typename T>
CPU_ONLY std::ostream& operator<<(std::ostream& out, const Matrix4<T>& m) {

	constexpr int width = 12;
	constexpr int precision = 5;

	out << "[\n";
	for (int i = 0; i < 4; ++i) {
		out << "  [";
		for (int j = 0; j < 4; ++j) {
			T v = m.elements[i][j];

			if (std::abs(v) >= 1e4 || (std::abs(v) > 0 && std::abs(v) < 1e-4)) {
				out << std::scientific;
			}
			else {
				out << std::fixed;
			}

			out << std::setw(width)
				<< std::setprecision(precision)
				<< v;

			if (j < 3) out << " ";
		}
		out << "]";
		if (i < 3) out << ",";
		out << "\n";
	}
	out << "]";
	return out;
}

#endif