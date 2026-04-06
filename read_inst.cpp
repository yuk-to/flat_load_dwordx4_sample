#include<iostream>
#include<memory>
#include<random>

#include<hip/hip_runtime.h>

#define check_status(X) \
	do { \
		if(X) { \
			std::cout << __LINE__ << ": " << hipGetErrorString(X) << std::endl; \
		} \
	} while(0)

__global__ void read_inst_kernel(
	const uint4* const a, // input
	uint4* const b, // output
	const size_t n, // number of elements to process
	const size_t mode = 0
) {
	// mode 0: normal load
	if(mode == 0)
	{
		for(auto i = 0; i < n; i++)
		{
			b[i] = a[i];
		}
	}
	// mode 1: load into a temporary variable before storing to output buffer
	else if(mode == 1)
	{
		for(auto i = 0; i < n; i++)
		{
			uint4 loaded;
			loaded = a[i];
			b[i] = loaded;
		}
	}
	// mode 2: use inline assembly to load the data, wait for load to complete before output
	else if(mode == 2)
	{
		for(auto i = 0; i < n; i++)
		{
			uint4 loaded;
			asm volatile(
				"flat_load_dwordx4 %0, %1 \n" 
				"s_waitcnt vmcnt(0) \n" // or s_wait_loadcnt(0)
				: "=v"(loaded) // uint32v4
				: "v"(a + i) // uint32v4*
				: "memory"
			);
			b[i] = loaded;
		}
	}
	// mode 3: use inline assembly to load the data, do not wait for load to complete before output
	// this cause incorrect results
	else if(mode == 3)
	{
		for(auto i = 0; i < n; i++)
		{
			uint4 loaded;
			asm volatile(
				"flat_load_dwordx4 %0, %1 \n" 
				: "=v"(loaded) // uint32v4
				: "v"(a + i) // uint32v4*
				: "memory"
			);
			b[i] = loaded;
		}
	}
}


using T=uint4;

int task(const size_t n, const size_t mode)
{
	auto a = std::make_unique<T[]>(n);
	auto b = std::make_unique<T[]>(n);

	// initialize inputs with random values
#if 0
	std::random_device seed_gen;
	auto seed = seed_gen();
#else
	auto seed = n; // fixed seed for reproducibility
#endif
	std::mt19937 engine(seed);
	for(auto i = 0; i < n; i++)
	{
		a[i].x = engine();
		a[i].y = engine();
		a[i].z = engine();
		a[i].w = engine();
	}

	// allocate and copy data to device
	T *dev_a, *dev_b;
	check_status(hipMalloc(&dev_a, sizeof(T) * n));
	check_status(hipMalloc(&dev_b, sizeof(T) * n));
	check_status(hipMemcpy(dev_a, a.get(), sizeof(T)*n, hipMemcpyHostToDevice));
	
	// launch the kernel
	read_inst_kernel<<<1,1>>>(dev_a, dev_b, n, mode);
	check_status(hipDeviceSynchronize());

	// copy data back to host and free device memory
	check_status(hipMemcpy(b.get(), dev_b, sizeof(T)*n, hipMemcpyDeviceToHost));
	check_status(hipDeviceSynchronize());
	
	int ret = 0;
	// verify the results
	for(auto i = 0; i < n; i++)
	{
		if(a[i] != b[i])
		{
			std::cout << std::hex;
			std::cout << i << ": "<< a[i].x << " != " << b[i].x << std::endl;
			std::cout << i << ": "<< a[i].y << " != " << b[i].y << std::endl;
			std::cout << i << ": "<< a[i].z << " != " << b[i].z << std::endl;
			std::cout << i << ": "<< a[i].w << " != " << b[i].w << std::endl;
			std::cout << std::dec;
			ret = 1;
		}
	}

	check_status(hipFree(dev_a));
	check_status(hipFree(dev_b));

	return ret;
}

int main(const int argc, const char * const * const argv) {
	if (argc == 3) {
		const size_t n = std::atoi(argv[1]);
		const size_t mode = std::atoi(argv[2]);
		task(n, mode);
	}
	else { // default test
		const size_t n = 2;
		{
			const size_t mode = 2;
			std::cout << "mode 2" << std::endl;
			if(!task(n, mode))
			{
				std::cout << "all pass! " << std::endl;
			}
		}
		{
			const size_t mode = 3;
			std::cout << "mode 3" << std::endl;
			if(!task(n, mode))
			{
				std::cout << "all pass!" << std::endl;
			}
		}
	}

	return 0;
}
