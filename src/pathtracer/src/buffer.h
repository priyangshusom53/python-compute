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
};


template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t start,
	size_t nElement
);

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst,
	size_t nElements
);

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst,
	size_t nElements
);

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t nElements
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
void StructuredBuffer<SType, type>::reserve(size_t size) {
	
	if (type == BufferType::CPU_BUFFER) {
		void* data = std::malloc(size * sizeof(SType));
		if (data == nullptr) {
			throw StructuredBufferRuntimeError(
				std::string("StructuredBuffer Error: Failed to allocate CPU buffer of size=") +=
				std::to_string(size)
			);
		}
		_capacity = size;
		size_t _sizeToCopy = std::min(_capacity, _size);
		if (_data != nullptr) {
			std::memcpy(data, _data, _sizeToCopy * sizeof(SType));
			std::free(_data);
		}
		_data = data;
		_size = _sizeToCopy;
	}
	else if (type == BufferType::GPU_BUFFER) {
		void* data;
		cudaError_t error = cudaMalloc(&data, size * sizeof(SType));
		if (error != cudaSuccess) {
			throw StructuredBufferRuntimeError(
				std::string("StructuredBuffer Error: Failed to allocate GPU buffer of size=") +=
				std::to_string(size) +
				std::string(", CUDA error: ") + std::string(cudaGetErrorString(error))
			);
		}
		_capacity = size;
		size_t _sizeToCopy = std::min(_capacity, _size);
		if (_data != nullptr) {
			error = cudaMemcpy(data, _data, _sizeToCopy * sizeof(SType), cudaMemcpyDeviceToDevice);
			if (error != cudaSuccess) {
				throw StructuredBufferRuntimeError(
					std::string("StructuredBuffer Error: Failed to allocate GPU buffer of size=") +=
					std::to_string(size) +
					std::string(", CUDA error: ") + std::string(cudaGetErrorString(error))
				);
			}
			cudaFree(_data);
		}
		_data = data;
		_size = _sizeToCopy;
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::push_back(const SType& elem) {
	if (type == BufferType::CPU_BUFFER) {
		if ((_capacity - _size) > 0) {
			static_cast<SType*>(_data)[_size++] = elem;
		}
		else {
			size_t size = 1;
			reserve(2 * (std::max(size, _size)));
			static_cast<SType*>(_data)[_size++] = elem;
		}
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: push_back not available for GPU memory");
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::push_back(SType&& elem) {
	if (type == BufferType::CPU_BUFFER) {
		if ((_capacity - _size) > 0) {
			static_cast<SType*>(_data)[_size++] = elem;
		}
		else {
			size_t size = 1;
			reserve(2 * (std::max(size, _size)));
			static_cast<SType*>(_data)[_size++] = elem;
		}
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: push_back not available for GPU memory");
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::resize(size_t newSize) {
	if (type == BufferType::CPU_BUFFER) {
		if (newSize <= _capacity) {
			if (_size > newSize) {
				_size = newSize;
			}
			else {
				for (int i = _size; i < newSize; ++i)
					static_cast<SType*>(_data)[i] = SType();
			}
		}
		else {
			_capacity = newSize;
			void* data = std::malloc(_capacity * sizeof(SType));
			if (_data != nullptr) {
				std::memcpy(data, _data, _size * sizeof(SType));
				std::free(_data);
			}
			for (int i = _size; i < newSize; ++i)
				static_cast<SType*>(data)[i] = SType();
			_data = data;
			_size = newSize;
		}
		
	}
	else if (type == BufferType::GPU_BUFFER) {
		throw StructuredBufferRuntimeError("StructuredBuffer Error: resize not available for GPU memory");
	}
}

template<typename SType, BufferType type>
void StructuredBuffer<SType, type>::assign(SType* ptr, size_t count) {
	if (type == BufferType::CPU_BUFFER) {
		if (_data != nullptr) {
			std::free(_data);
			_size = 0;
			_capacity = 0;
		}
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
				(&data[i])->~SType();
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
void CopyData(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t start,
	size_t nElements
) {
	size_t bytes = nElements * sizeof(SType);
	if (nElements > dst.capacity() - dst.size()) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Not enough space in destination CPU buffer" +
			std::string(", requested count ") + std::to_string(nElements) +
			std::string(", available count ") + std::to_string(dst.capacity() - dst.size())
		);
	}
	size_t _sizeToCopy = nElements;
	SType* dstPtr = &dst[0] + dst.size();
	SType* srcPtr = &dst[0] + start;
	std::memcpy((void*)dstPtr, (void*)srcPtr, _sizeToCopy*sizeof(SType));
}

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst,
	size_t nElements
) {
	size_t bytes = nElements * sizeof(SType);
	if (bytes > dst.totalSize - dst.currentSize) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Not enough space in destination GPU buffer" +
			std::string(", requested ") + std::to_string(bytes) +
			std::string(", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	cudaError_t error = cudaMemcpy((void*)dstPtr, src.data, bytes, cudaMemcpyDeviceToDevice);
	if (error != cudaSuccess) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Failed to copy GPU to GPU buffer of size=" +
			std::to_string(bytes) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
	dst.currentSize += bytes;
}

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::CPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::GPU_BUFFER>& dst,
	size_t nElements
) {
	size_t bytes = nElements * sizeof(SType);
	if (bytes > dst.totalSize - dst.currentSize) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Not enough space in destination GPU buffer" +
			std::string(", requested ") + std::to_string(bytes) +
			std::string(", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	cudaError_t error = cudaMemcpy((void*)dstPtr, src.data, bytes, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Failed to copy CPU to GPU buffer of size=" +
			std::to_string(bytes) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
	dst.currentSize += bytes;
}

template<typename SType>
void CopyData(
	const StructuredBuffer<SType, BufferType::GPU_BUFFER>& src,
	StructuredBuffer<SType, BufferType::CPU_BUFFER>& dst,
	size_t nElements
) {
	size_t bytes = nElements * sizeof(SType);
	if (bytes > dst.totalSize - dst.currentSize) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Not enough space in destination CPU buffer" +
			std::string(", requested ") + std::to_string(bytes) +
			std::string(", available ") + std::to_string(dst.totalSize - dst.currentSize)
		);
	}
	char* dstPtr = static_cast<char*>(dst.data) + dst.currentSize;
	cudaError_t error = cudaMemcpy((void*)dstPtr, src.data, bytes, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		throw StructuredBufferRuntimeError(
			"StructuredBuffer Error: Failed to copy GPU to CPU buffer of size=" +
			std::to_string(bytes) +
			", CUDA error: " +
			std::string(cudaGetErrorString(error))
		);
	}
	dst.currentSize += bytes;
}

#pragma endregion

#endif