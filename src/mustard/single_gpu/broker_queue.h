
#include <cuda_runtime.h>

#pragma once

class BrokerWorkDistributor
{
public:
	typedef unsigned int Ticket;
	typedef unsigned int HT_t;
	typedef unsigned long long int HT;

	volatile Ticket *tickets;
	unsigned int *ring_buffer;
    int N;

	HT *head_tail;
	int *count;

    BrokerWorkDistributor(int size)
    {
        N = size;
        cudaMalloc(&tickets, sizeof(Ticket) * N);
        cudaMalloc(&ring_buffer, sizeof(unsigned int) * N);
        cudaMalloc(&count, sizeof(int));
        cudaMalloc(&head_tail, sizeof(HT));
    }

    void free_mem()
    {
        cudaFree((void *)tickets);
        cudaFree((void *)ring_buffer);
        cudaFree((void *)count);
        cudaFree((void *)head_tail);
    }

	__device__ static __forceinline__ void backoff()
	{
		__threadfence();
	}

	template <typename L>
	__device__ static __forceinline__ L uncachedLoad(const L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ static __forceinline__ L atomicLoad(const L* l)
	{
		return *l;
	}

	__device__ HT_t* head(HT* head_tail)
	{
		return reinterpret_cast<HT_t*>(head_tail) + 1;
	}

	__device__ HT_t* tail(HT* head_tail)
	{
		return reinterpret_cast<HT_t*>(head_tail);
	}

	__forceinline__ __device__ void waitForTicket(const unsigned int P, const Ticket number)
	{
		while (tickets[P] != number)
		{
			backoff(); // back off
		}
	}

	__forceinline__ __device__ bool ensureDequeue()
	{
		int Num = atomicLoad(count);
		bool ensurance = false;

		while (!ensurance && Num > 0)
		{
			if (atomicSub(count, 1) > 0)
			{
				ensurance = true;
			}
			else
			{
				Num = atomicAdd(count, 1) + 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ bool ensureEnqueue()
	{
		int Num = atomicLoad(count);
		bool ensurance = false;

		while (!ensurance && Num < N)
		{
			if (atomicAdd(count, 1) < N)
			{
				ensurance = true;
			}
			else
			{
				Num = atomicSub(count, 1) - 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ void readData(unsigned int& val)
	{
		const unsigned int Pos = atomicAdd(head(const_cast<HT*>(head_tail)), 1);
		const unsigned int P = Pos % N;

		waitForTicket(P, 2 * (Pos / N) + 1);
		val = ring_buffer[P];
		__threadfence();
		tickets[P] = 2 * ((Pos + N) / N);
	}

	__forceinline__ __device__ void putData(const unsigned int data)
	{
		const unsigned int Pos = atomicAdd(tail(const_cast<HT*>(head_tail)), 1);
		const unsigned int P = Pos % N;
		const unsigned int B = 2 * (Pos / N);

		waitForTicket(P, B);
		ring_buffer[P] = data;
		__threadfence();
		tickets[P] = B + 1;
	}

	__device__ void init()
	{
		const int lid = threadIdx.x + blockIdx.x * blockDim.x;

		if (lid == 0)
		{
			*count = 0;
			*head_tail = 0x0ULL;
		}

		for (int v = lid; v < N; v += blockDim.x * gridDim.x)
		{
			// ring_buffer[v] = T(0x0);
			tickets[v] = 0x0;
		}
	}

	__device__ inline bool enqueue(const unsigned int & data)
	{
		bool writeData = ensureEnqueue();
		if (writeData)
		{
			putData(data);
		} else {
        }
		return false;
	}

	__device__ inline void dequeue(bool& hasData, unsigned int & data)
	{
		hasData = ensureDequeue();
		if (hasData)
		{
			readData(data);
		}
	}

	__device__ int size() const
	{
		return atomicLoad(count);
	}
};