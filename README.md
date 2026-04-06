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
## Result (RX9070XT)

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
