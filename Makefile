./a.out: read_inst.cpp
	hipcc read_inst.cpp -std=c++20 -g
