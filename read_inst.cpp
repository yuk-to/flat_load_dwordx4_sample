#include<iostream>
#include<memory>
#include<random>

#include<hip/hip_runtime.h>

// flat_load_dwordx4 requires synchronization, so we need to wait for the load to complete

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
#if 0
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
	// mode 5, unroll 16 loads
	else if(mode == 4) // Issueing 15 flat_load_dwordx4 instructions, encounted illegal memory access.
	{
		auto i = 0;
		for(; i + 16 <= n; i+=16)
		{
			uint4 loaded[16];
			asm volatile(
				// "flat_load_dwordx4 %0, %16 \n"
				"flat_load_dwordx4 %1, %17 \n"
				"flat_load_dwordx4 %2, %18 \n"
				"flat_load_dwordx4 %3, %19 \n"
				"flat_load_dwordx4 %4, %20 \n"
				"flat_load_dwordx4 %5, %21 \n"
				"flat_load_dwordx4 %6, %22 \n"
				"flat_load_dwordx4 %7, %23 \n"
				"flat_load_dwordx4 %8, %24 \n"
				"flat_load_dwordx4 %9, %25 \n"
				"flat_load_dwordx4 %10, %26 \n"
				"flat_load_dwordx4 %11, %27 \n"
				"flat_load_dwordx4 %12, %28 \n"
				"flat_load_dwordx4 %13, %29 \n"
				"flat_load_dwordx4 %14, %30 \n"
				"flat_load_dwordx4 %15, %31 \n"
				"s_waitcnt vmcnt(0) \n" // or s_wait_loadcnt(0)
				: "=v"(loaded[0]), "=v"(loaded[1]), "=v"(loaded[2]), "=v"(loaded[3]), "=v"(loaded[4]), "=v"(loaded[5]), "=v"(loaded[6]), "=v"(loaded[7]),
				  "=v"(loaded[8]), "=v"(loaded[9]), "=v"(loaded[10]), "=v"(loaded[11]), "=v"(loaded[12]), "=v"(loaded[13]), "=v"(loaded[14]), "=v"(loaded[15])
				: "v"(a + i + 0), "v"(a + i + 1), "v"(a + i + 2), "v"(a + i + 3), "v"(a + i + 4), "v"(a + i + 5), "v"(a + i + 6), "v"(a + i + 7),
				  "v"(a + i + 8), "v"(a + i + 9), "v"(a + i + 10), "v"(a + i + 11), "v"(a + i + 12), "v"(a + i + 13), "v"(a + i + 14), "v"(a + i + 15)
				: "memory"
			);
			b[i+0 ] = loaded[0 ];
			b[i+1 ] = loaded[1 ];
			b[i+2 ] = loaded[2 ];
			b[i+3 ] = loaded[3 ];
			b[i+4 ] = loaded[4 ];
			b[i+5 ] = loaded[5 ];
			b[i+6 ] = loaded[6 ];
			b[i+7 ] = loaded[7 ];
			b[i+8 ] = loaded[8 ];
			b[i+9 ] = loaded[9 ];
			b[i+10] = loaded[10];
			b[i+11] = loaded[11];
			b[i+12] = loaded[12];
			b[i+13] = loaded[13];
			b[i+14] = loaded[14];
			b[i+15] = loaded[15];
		}
		for(; i < n; i++)
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
	// mode 5, unroll 17 loads
	else if(mode == 5)
	{
		auto i = 0;
		for(; i + 17 <= n; i+=17)
		{
			uint4 loaded[17];
			asm volatile(
				"flat_load_dwordx4 %0,  %17 \n"
				"flat_load_dwordx4 %1,  %18 \n"
				"flat_load_dwordx4 %2,  %19 \n"
				"flat_load_dwordx4 %3,  %20 \n"
				"flat_load_dwordx4 %4,  %21 \n"
				"flat_load_dwordx4 %5,  %22 \n"
				"flat_load_dwordx4 %6,  %23 \n"
				"flat_load_dwordx4 %7,  %24 \n"
				"flat_load_dwordx4 %8,  %25 \n"
				"flat_load_dwordx4 %9,  %26 \n"
				"flat_load_dwordx4 %10, %27 \n"
				"flat_load_dwordx4 %11, %28 \n"
				"flat_load_dwordx4 %12, %29 \n"
				"flat_load_dwordx4 %13, %30 \n"
				"flat_load_dwordx4 %14, %31 \n"
				"flat_load_dwordx4 %15, %32 \n"
				"flat_load_dwordx4 %16, %33 \n"
				"s_waitcnt vmcnt(0) \n" // or s_wait_loadcnt(0)
				: "=v"(loaded[0]), "=v"(loaded[1]), "=v"(loaded[2]), "=v"(loaded[3]), "=v"(loaded[4]), "=v"(loaded[5]), "=v"(loaded[6]), "=v"(loaded[7]),
				  "=v"(loaded[8]), "=v"(loaded[9]), "=v"(loaded[10]), "=v"(loaded[11]), "=v"(loaded[12]), "=v"(loaded[13]), "=v"(loaded[14]), "=v"(loaded[15]), "=v"(loaded[16])
				: "v"(a + i + 0), "v"(a + i + 1), "v"(a + i + 2), "v"(a + i + 3), "v"(a + i + 4), "v"(a + i + 5), "v"(a + i + 6), "v"(a + i + 7),
				  "v"(a + i + 8), "v"(a + i + 9), "v"(a + i + 10), "v"(a + i + 11), "v"(a + i + 12), "v"(a + i + 13), "v"(a + i + 14), "v"(a + i + 15), "v"(a + i + 16)
				: "memory"
			);
			b[i+0 ] = loaded[0 ];
			b[i+1 ] = loaded[1 ];
			b[i+2 ] = loaded[2 ];
			b[i+3 ] = loaded[3 ];
			b[i+4 ] = loaded[4 ];
			b[i+5 ] = loaded[5 ];
			b[i+6 ] = loaded[6 ];
			b[i+7 ] = loaded[7 ];
			b[i+8 ] = loaded[8 ];
			b[i+9 ] = loaded[9 ];
			b[i+10] = loaded[10];
			b[i+11] = loaded[11];
			b[i+12] = loaded[12];
			b[i+13] = loaded[13];
			b[i+14] = loaded[14];
			b[i+15] = loaded[15];
			b[i+16] = loaded[16];
		}
		for(; i < n; i++)
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
	else if(mode == 6)
	{
		for(auto i = 0; i < n; i++)
		{
			uint4 loaded;
			asm volatile(
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				// "s_waitcnt vmcnt(0) \n" // or s_wait_loadcnt(0)
				: "=v"(loaded) // uint32v4
				: "v"(a + i) // uint32v4*
				: "memory"
			);
			b[i] = loaded;
			asm volatile(
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				// "s_waitcnt vmcnt(0) \n" // or s_wait_loadcnt(0)
				: "=v"(loaded) // uint32v4
				: "v"(a + i) // uint32v4*
				: "memory"
			);
			b[i] = loaded;
		}
	}
	else if(mode == 7)
	{
		for(auto i = 0; i < n; i++)
		{
			uint4 loaded;
			asm volatile(
				"flat_load_dwordx4 %0, %8 \n" 
				"flat_load_dwordx4 %0, %7 \n" 
				"flat_load_dwordx4 %0, %6 \n" 
				"flat_load_dwordx4 %0, %5 \n" 
				"flat_load_dwordx4 %0, %4 \n" 
				"flat_load_dwordx4 %0, %3 \n" 
				"flat_load_dwordx4 %0, %2 \n" 
				"flat_load_dwordx4 %0, %1 \n" 
				"s_waitcnt vmcnt(0) \n" // or s_wait_loadcnt(0)
				: "=v"(loaded) // uint32v4
				: "v"(a + i + 0), "v"(a + i + 1), "v"(a + i + 2), "v"(a + i + 3), "v"(a + i + 4), "v"(a + i + 5), "v"(a + i + 5), "v"(a + i + 7)
				: "memory"
			);
			b[i] = loaded;
		}
	}
	else if(mode == 8)
	{
		auto i = 0;
		constexpr int unroll_count = 64;
		for(; i + unroll_count <= n; i+=unroll_count)
		{
			uint4 loaded[unroll_count];
			for(int j = 0; j < unroll_count; j++)
			{
				asm volatile(
					"flat_load_dwordx4 %0,  %1 \n"
					: "=v"(loaded[i + j])
					: "v"(a + i + j)
					: "memory"
				);
			}
			// asm volatile("s_waitcnt vmcnt(0) \n":::);
			for(int j = 0; j < unroll_count; j++)
			{
				b[i + j] = loaded[i + j];
			}
		}
		for(; i < n; i++)
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
	else if(mode == 9)
	{
		auto i = 0;
		constexpr int unroll_count = 128;
		// constexpr int unroll_count = 16;
		for(auto i = 0; i < n; i++)
		{
			uint4 loaded;
			const auto input = a + i;
			for(int j = 0; j < unroll_count; j++)
			{
				asm volatile(
					"flat_load_dwordx4 %0,  %1 \n"
					: "=v"(loaded)
					: "v"(input)
					: "memory"
				);
			}
			asm volatile("s_waitcnt vmcnt(0) \n":::);
			b[i] = loaded;
		}
	}
#endif
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

__global__ void read_inst_kernel_2(
	const uint4* const a, // input
	uint4* const b, // output
	const size_t n // number of elements to process
) {
	for(auto i = 0; i < n; i++)
	{
		uint4 loaded;
		asm volatile(
			"flat_load_dwordx4 %0, %1 \n" 
			: "=&v"(loaded) // uint32v4
			: "v"(a + i) // uint32v4*
			: "memory"
		);
		// asm volatile("s_waitcnt vmcnt(0) \n":::);
		b[i] = loaded;
	}
}

int task2(const size_t n)
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
	printf("%p %p %ld\n", dev_a, dev_b, n);
	read_inst_kernel_2<<<1,1>>>(dev_a, dev_b, n);
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
		if(mode == 100)
		{
			task2(n);
		}
		else
		{
			task(n, mode);
		}
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
		{
			const size_t n = 16;
			const size_t mode = 4;
			std::cout << "mode 4" << std::endl;
			if(!task(n, mode))
			{
				std::cout << "all pass!" << std::endl;
			}
		}
		{
			const size_t n = 17;
			const size_t mode = 5;
			std::cout << "mode 5" << std::endl;
			if(!task(n, mode))
			{
				std::cout << "all pass!" << std::endl;
			}
		}
	}

	return 0;
}
