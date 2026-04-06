## Prerequirements
* Linux
* HIP
* Make

## Usage

### build & run
```bash
$ make && ./a.out
```

### options
```bash
$ ./a.out DataSize Mode
```
## Results (RX9070XT)

### default

```bash
yuk-to@Deneb ~/flat_load_dwordx4_sample % rocm-smi


WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status

======================================= ROCm System Management Interface =======================================
================================================= Concise Info =================================================
Device  Node  IDs              Temp    Power  Partitions          SCLK   MCLK   Fan  Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)
================================================================================================================
0       1     0x7550,   45807  32.0_C  17.0W  N/A, N/A, 0         41Mhz  96Mhz  0%   auto  304.0W  82%    3%
================================================================================================================
============================================= End of ROCm SMI Log ==============================================
yuk-to@Deneb ~/flat_load_dwordx4_sample % ./a.out
mode 2
all pass!
mode 3
0: 6f9d5ca8 != a4c00000
0: 2f618a0f != 7c24
0: 6a319ed != 0
0: ee797648 != 1
1: 8cb6ae16 != a4c00010
1: f29e792b != 7c24
1: 6f7149d2 != 0
1: 7c18844b != 1
```

### Illegal Memory Access

```
yuk_to@Deneb ~/flat_load_dwordx4_sample % ./a.out 15 3 2&>1|head
0: d94c1dc8 != 69200000
0: d046cef5 != 7988
0: 2dcc1f8c != bbda16ed
0: 97630a85 != 0
1: deabf77 != 69200010
1: 1c699980 != 7988
1: 5c8dc89c != bbda16ed
1: d1908a9b != 0
2: 4680acc7 != 69200020
2: ed3876cb != 7988
yuk_to@Deneb ~/flat_load_dwordx4_sample % ./a.out 16 3 2&>1|head
Kernel Name: _Z16read_inst_kernelPK15HIP_vector_typeIjLj4EEPS0_mm
VGPU=0x4b2a80 SWq=0x7ae4f8970000, HWq=0x7ae4f5700000, id=1
        Dispatch Header =0xd02 (type=2, barrier=1, acquire=2, release=1), setup=0
        grid=[1, 1, 1], workgroup=[1, 1, 1]
        private_seg_size=0, group_seg_size=0
        kernel_obj=0x7ae4f8940680, kernarg_address=0x0x7ae4f5200100
        completion_signal=0x0, correlation_id=0
        rptr=1, wptr=3
 104: an illegal memory access was encountered
107: an illegal memory access was encountered
```
