#ifndef BUFFER_H
#define BUFFER_H

#include "cudadefines.h"

#include<cuda_runtime_api.h >

#include <cstdlib>
#include <stdexcept>
#include <string>


class AttributeBufferRuntimeError : public std::runtime_error {
	public:
		AttributeBufferRuntimeError(const std::string& message) : std::runtime_error(message) {}
};

class StructuredBufferRuntimeError : public std::runtime_error {
	public:
		StructuredBufferRuntimeError(const std::string& message) : std::runtime_error(message) {}
};

enum BufferType {
	CPU_BUFFER = 0,
	GPU_BUFFER = 1
};

//enum CopyDirection {
//	CPU_TO_CPU = 0,
//	GPU_TO_GPU = 1,
//	CPU_TO_GPU = 2,
//	GPU_TO_CPU = 3
//};

template<typename T, BufferType type>
struct AttributeBuffer {
	void* data = nullptr;
	int nDims;					// max 3
	int shape[3];
	size_t totalSize = 0;
	size_t currentSize = 0;		// currently filled size in bytes
	int bufferMemoryLocation = type;
	AttributeBuffer();
	AttributeBuffer(int nDims, const int shape[3]);
	void Allocate();
	~AttributeBuffer();
};

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::CPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::CPU_BUFFER>& dst,
	size_t nElements);

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::GPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::GPU_BUFFER>& dst,
	size_t nElements);

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::CPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::GPU_BUFFER>& dst,
	size_t nElements);

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::GPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::CPU_BUFFER>& dst,
	size_t nElements);


template<typename SType>
struct StructuredBufferView;

template<typename SType, BufferType type>
class StructuredBuffer {
	void* _data;
	size_t _size = 0;			// number of elements currently stored
	size_t _capacity = 0;		
public:
	StructuredBuffer();
	StructuredBuffer(size_t elemCount);
	size_t size() const;
	size_t capacity() const;
	SType* data();
	const SType* data() const;
	void reserve(size_t size);
	void push_back(const SType& elem);
	void push_back(SType&& elem);
	void resize(size_t newSize);
	void assign(SType* ptr, size_t count);
	SType& operator[](int idx);
	const SType& operator[](int idx) const;
	StructuredBufferView<SType> view() const;
	bool empty() const;
	void clear();
	~StructuredBuffer();
	static_assert(
		std::is_move_constructible_v<SType>,
		"StructuredBuffer requires move-constructible types"
		);
	static_assert(
		std::is_trivially_copyable_v<SType>,
		"GPU StructuredBuffer only supports trivially copyable types"
		);
};


template<typename SType>
void push_back_array(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t start,	// starting index in src buffer
	size_t nElement
);

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst
);

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst
);

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst
);

template<typename SType>
struct StructuredBufferView {
	SType* data;
	size_t length;

	CPU_GPU SType& operator[](int idx) {
		return data[idx];
	}

	CPU_GPU const SType& operator[](int idx) const {
		return data[idx];
	}
};


// AttributeBuffer definition
template<typename T, BufferType type>
AttributeBuffer<T, type>::AttributeBuffer() : data(nullptr), nDims(0), totalSize(0), currentSize(0){
	shape[0] = shape[1] = shape[2] = 0;
}

template<typename T, BufferType type>
AttributeBuffer<T, type>::AttributeBuffer(int nDims, const int shape[3]) : nDims(nDims) {
	if (nDims == 1) {
		this->shape[0] = shape[0];
		this->shape[1] = 1;
		this->shape[2] = 1;
	}
	else if (nDims == 2) {
		this->shape[0] = shape[0];
		this->shape[1] = shape[1];
		this->shape[2] = 1;
	}
	else if (nDims == 3) {
		this->shape[0] = shape[0];
		this->shape[1] = shape[1];
		this->shape[2] = shape[2];
	}
}

template<typename T, BufferType type>
void AttributeBuffer<T, type>::Allocate() {
	size_t size = shape[0] * shape[1] * shape[2] * sizeof(T);
	if (type == BufferType::CPU_BUFFER) {
		data = std::malloc(size);
		if(data == nullptr){
			throw AttributeBufferRuntimeError(
				std::string("AttributeBuffer Error: Failed to allocate CPU buffer of size=") +=
				std::to_string(size)
				);
		}
		totalSize = size;
	}
	else if (type == BufferType::GPU_BUFFER) {
		cudaError_t error = cudaMalloc(&data, size);
		if (error != cudaSuccess) {
			throw AttributeBufferRuntimeError(
				std::string("AttributeBuffer Error: Failed to allocate GPU buffer of size=") +=
				std::to_string(size) +
				std::string(", CUDA error: ") + std::string(cudaGetErrorString(error)));
		}
		totalSize = size;
	}
}

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::CPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::CPU_BUFFER>& dst,
	size_t nElements) {

	size_t bytes = nElements * sizeof(T);
	if (bytes > (dst.totalSize - dst.currentSize)) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Not enough space in destination CPU buffer" + 
			std::string(", requested ") + std::to_string(bytes) +
			std::string(", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	std::memcpy((void*)dstPtr, src.data, bytes);
	dst.currentSize += bytes;
}

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::GPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::GPU_BUFFER>& dst,
	size_t nElements) {

	size_t bytes = nElements * sizeof(T);
	if (bytes > (dst.totalSize - dst.currentSize)) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Not enough space in destination GPU buffer" +
			std::string(", requested ") + std::to_string(bytes) +
			std::string (", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	cudaError_t error = cudaMemcpy((void*)dstPtr, src.data, bytes, cudaMemcpyDeviceToDevice);
	if (error != cudaSuccess) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Failed to copy GPU to GPU buffer of size=" +
			std::to_string(bytes) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
	dst.currentSize += bytes;
}

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::CPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::GPU_BUFFER>& dst,
	size_t nElements) {
	size_t bytes = nElements * sizeof(T);
	if (bytes > (dst.totalSize - dst.currentSize)) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Not enough space in destination GPU buffer" +
			std::string(", requested ") + std::to_string(bytes) +
			std::string(", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	cudaError_t error = cudaMemcpy((void*)dstPtr, src.data, bytes, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Failed to copy CPU to GPU buffer of size=" +
			std::to_string(bytes) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
	dst.currentSize += bytes;
}

template<typename T>
void CopyData(
	const AttributeBuffer<T, BufferType::GPU_BUFFER>& src,
	AttributeBuffer<T, BufferType::CPU_BUFFER>& dst,
	size_t nElements) {
	size_t bytes = nElements * sizeof(T);
	if (bytes > (dst.totalSize - dst.currentSize)) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Not enough space in destination CPU buffer" +
			std::string(", requested ") + std::to_string(bytes) +
			std::string(", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	cudaError_t error = cudaMemcpy((void*)dstPtr, src.data, bytes, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		throw AttributeBufferRuntimeError(
			"AttributeBuffer Error: Failed to copy GPU to CPU buffer of size=" +
			std::to_string(bytes) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
	dst.currentSize += bytes;
}

template<typename T, BufferType type>
AttributeBuffer<T, type>::~AttributeBuffer() {
	if (data != nullptr) {
		if (type == BufferType::CPU_BUFFER) {
			std::free(data);
		}
		else if (type == BufferType::GPU_BUFFER) {
			cudaFree(data);
		}
	}
}

#pragma region StructuredBuffer DEFINITION

template<typename SType, BufferType type>
StructuredBuffer<SType, type>::StructuredBuffer() : _data(nullptr), _size(0),_capacity(0)  {}

template<typename SType, BufferType type>
StructuredBuffer<SType, type>::StructuredBuffer(size_t size) : _data(nullptr), _size(0), _capacity(0){
	_capacity = size;
	_size = size;
	if (type == BufferType::CPU_BUFFER) {
		_data = std::malloc(_size * sizeof(SType));
		if(_data == nullptr)
			throw StructuredBufferRuntimeError("StructuredBuffer Error: failed to allocate CPU memory");
		SType* data = static_cast<SType*>(_data);
		for (int i = 0; i < _size; ++i)
			new (&data[i]) SType();
	}
	else if (type == BufferType::GPU_BUFFER) {
		cudaError_t err = cudaMalloc(&_data, _size * sizeof(SType));
		if (err != cudaSuccess)
			throw StructuredBufferRuntimeError(cudaGetErrorString(err));
	}
}

template<typename SType, BufferType type>
size_t StructuredBuffer<SType, type>::size() const {
	return _size;
}

template<typename SType, BufferType type>
size_t StructuredBuffer<SType, type>::capacity() const {
	return _capacity;
}

template<typename SType, BufferType type>
SType* StructuredBuffer<SType, type>::data() {
	return static_cast<SType*>(_data);
}

template<typename SType, BufferType type>
const SType* StructuredBuffer<SType, type>::data() const {
	return static_cast<SType*>(_data);
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::reserve(size_t newCapacity) {
	
	if (newCapacity <= _capacity)
		return;

	if (type == BufferType::CPU_BUFFER) {
		void* data = std::malloc(newCapacity * sizeof(SType));
		if (data == nullptr) {
			throw StructuredBufferRuntimeError(
				std::string("StructuredBuffer Error: Failed to allocate CPU buffer of size=") +=
				std::to_string(newCapacity)
			);
		}
		size_t _sizeToCopy = _size;
		if (_data != nullptr) {
			if(std::is_trivially_copyable_v<SType>)
				std::memcpy(data, _data, _sizeToCopy * sizeof(SType));
			else {
				SType* _tdata = static_cast<SType*>(_data);
				SType* tdata = static_cast<SType*>(data);
				for (int i = 0; i < _sizeToCopy; ++i) {
					new (&tdata[i]) SType(std::move(_tdata[i]));
					_tdata[i].~SType();
				}
			}
			std::free(_data);
		}
		_data = data;
		_capacity = newCapacity;
	}
	else if (type == BufferType::GPU_BUFFER) {
		void* data;
		cudaError_t error = cudaMalloc(&data, newCapacity * sizeof(SType));
		if (error != cudaSuccess) {
			throw StructuredBufferRuntimeError(
				std::string("StructuredBuffer Error: Failed to allocate GPU buffer of size=") +=
				std::to_string(newCapacity) +
				std::string(", CUDA error: ") + std::string(cudaGetErrorString(error))
			);
		}
		size_t _sizeToCopy = _size;
		if (_data != nullptr) {
			error = cudaMemcpy(data, _data, _sizeToCopy * sizeof(SType), cudaMemcpyDeviceToDevice);
			if (error != cudaSuccess) {
				cudaFree(data);
				throw StructuredBufferRuntimeError(
					std::string("StructuredBuffer Error: Failed to copy GPU buffer of size=") +=
					std::to_string(_sizeToCopy) +
					std::string(", CUDA error: ") + std::string(cudaGetErrorString(error))
				);
			}
			cudaFree(_data);
		}
		_data = data;
		_capacity = newCapacity;
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::push_back(const SType& elem) {
	if (type == BufferType::CPU_BUFFER) {
		if (_size == _capacity) {
			size_t newCap = (_capacity == 0) ? 2 : _capacity * 2;
			reserve(newCap);
		}

		SType* data = static_cast<SType*>(_data);
		new (&data[_size]) SType(elem); 
		++_size;
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: push_back not available for GPU memory");
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::push_back(SType&& elem) {
	if (type == BufferType::CPU_BUFFER) {
		if (_size == _capacity) {
			size_t newCap = (_capacity == 0) ? 2 : _capacity * 2;
			reserve(newCap);
		}

		SType* data = static_cast<SType*>(_data);
		new (&data[_size]) SType(std::move(elem));
		++_size;
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: push_back not available for GPU memory");
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::resize(size_t newSize) {
	if (type == BufferType::CPU_BUFFER) {
		SType* data = static_cast<SType*>(_data);
		if (newSize < _size) {
			for (size_t i = newSize; i < _size; ++i) {
				data[i].~SType();
			}
			_size = newSize;
			return;
		}
		if (newSize <= _capacity) {
			for (size_t i = _size; i < newSize; ++i) {
				new (&data[i]) SType();
			}
			_size = newSize;
			return;
		}
		size_t newCapacity = newSize;
		SType* newData = static_cast<SType*>(
			std::malloc(newCapacity * sizeof(SType))
			);
		if (!newData) {
			throw StructuredBufferRuntimeError("Allocation failed");
		}

		// MOVE-construct old elements
		for (size_t i = 0; i < _size; ++i) {
			new (&newData[i]) SType(std::move(data[i]));
			data[i].~SType();
		}

		// Default-construct new elements
		for (size_t i = _size; i < newSize; ++i) {
			new (&newData[i]) SType();
		}

		std::free(_data);
		_data = (void*)newData;
		_capacity = newCapacity;
		_size = newSize;
		
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: resize not available for GPU memory");
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::assign(SType* ptr, size_t count) {
	if (type == BufferType::CPU_BUFFER) {
		clear();
		std::free(_data);
		_data = static_cast<void*>(ptr);
		_size = count;
		_capacity = count;
	}
	else if (type == BufferType::GPU_BUFFER) {
		if (_data != nullptr) {
			cudaFree(_data);
			_size = 0;
			_capacity = 0;
		}
		_data = static_cast<void*>(ptr);
		_size = count;
		_capacity = count;
	}
}

template<typename SType, BufferType type>
SType& StructuredBuffer<SType, type>::operator[](int idx) {
	if (type == BufferType::CPU_BUFFER) {
#ifdef DEBUG
		if (idx < 0 || idx >= _size)
			throw StructuredBufferRuntimeError("StructuredBuffer Error: Index out of bound");
#endif
		return *(static_cast<SType*>(_data) + idx);
	}
	else if (type == BufferType::GPU_BUFFER)
		throw StructuredBufferRuntimeError("StructuredBuffer Error: StructuredBuffer indexing not available for GPU memory\nGPU memory indexing available through StructuredBufferView");
}

template<typename SType, BufferType type>
const SType& StructuredBuffer<SType, type>::operator[](int idx) const {
	if (type == BufferType::CPU_BUFFER) {
#ifdef DEBUG
		if (idx < 0 || idx >= _size)
			throw StructuredBufferRuntimeError("StructuredBuffer Error: Index out of bound");
#endif
		return *(static_cast<SType*>(_data) + idx);
	}
	else if(type == BufferType::GPU_BUFFER)
		throw StructuredBufferRuntimeError("StructuredBuffer Error: StructuredBuffer indexing not available for GPU memory\nGPU memory indexing available through StructuredBufferView");
}

template<typename SType, BufferType type>
StructuredBufferView<SType> StructuredBuffer<SType, type>::view() const {
	return { static_cast<SType*>(_data),_size };
}

template<typename SType, BufferType type>
bool StructuredBuffer<SType, type>::empty() const {
	return _size == 0;
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::clear() {
	if (type == BufferType::CPU_BUFFER) {
		if (_data != nullptr) {
			SType* data = static_cast<SType*>(_data);
			for (int i = 0; i < _size; ++i) {
				data[i].~SType();
			}
		}
		_size = 0;
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: clear not available for GPU memory");
	}
}

template<typename SType, BufferType type>
StructuredBuffer<SType, type>::~StructuredBuffer() {
	if (type == BufferType::CPU_BUFFER) {
		if (_data) {
			SType* data = static_cast<SType*>(_data);
			for (size_t i = 0; i < _size; ++i) {
				data[i].~SType();
			}
			std::free(_data);
		}
	}
	else if (type == BufferType::GPU_BUFFER)
		cudaFree(_data);
	_size = 0;
	_capacity = 0;
}

template<typename SType>
void push_back_array(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t start,
	size_t nElements
) {
	if (start + nElements > src.size()) {
		throw StructuredBufferRuntimeError("Source range out of bounds");
	}

	size_t oldSize = dst.size();
	dst.resize(oldSize + nElements);

	SType* dstPtr = dst.data() + oldSize;
	const SType* srcPtr = src.data() + start;

	if constexpr (std::is_trivially_copyable_v<SType>) {
		std::memcpy(dstPtr, srcPtr, nElements * sizeof(SType));
	}
	else {
		for (size_t i = 0; i < nElements; ++i) {
			new (&dstPtr[i]) SType(srcPtr[i]); // copy construct
		}
	}
}

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst
) {
	if (src.size() != dst.size()) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: GPU src and dst buffer size must be same for copying");
	}
	const SType* d_srcPtr = src.data();
	SType* d_dstPtr = dst.data();
	cudaError_t error = cudaMemcpy((void*)d_dstPtr, (void*)d_srcPtr, (src.size()) * sizeof(SType), cudaMemcpyDeviceToDevice);
	if (error != cudaSuccess) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Failed to copy GPU to GPU buffer of size=" +
			std::to_string(src.size()) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
}

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst
) {
	if (src.size() != dst.size()) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: CPU src and GPU dst buffer size must be same for copying");
	}
	const SType* srcPtr = &src[0];
	SType* d_dstPtr = dst.data();
	cudaError_t error = cudaMemcpy((void*)d_dstPtr, (void*)srcPtr, (src.size()) * sizeof(SType), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Failed to copy CPU to GPU buffer of size=" +
			std::to_string(src.size()) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
}

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t nElement
) {
	if (src.size() != dst.size()) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: GPU src and CPU dst buffer size must be same for copying");
	}
	const SType* d_srcPtr = src.data();
	SType* dstPtr = &dst[0];
	cudaError_t error = cudaMemcpy((void*)dstPtr, (void*)d_srcPtr, (src.size()) * sizeof(SType), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Failed to copy GPU to CPU buffer of size=" +
			std::to_string(src.size()) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
}

#pragma endregion

#endif