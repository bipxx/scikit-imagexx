# Boolean and signed conversion performance

Exact values, dtype, warnings, identity and input immutability are checked before timing. Fallback is disabled. Reference pools use one thread; native worker limits are explicit. All raw samples and adverse cases are retained.

| API | Input | Workload | Workers | Execution | Original ms | Current ms | Original/current | Preserved/current |
|---|---|---|---:|---|---:|---:|---:|---:|
| util.img_as_bool | bool | contiguous | 1 | identity | 0.0006 | 0.0003 | 1.8513x | 1.0137x |
| util.img_as_bool | bool | fortran | 1 | identity | 0.0006 | 0.0003 | 1.7491x | 0.9971x |
| util.img_as_bool | bool | reversed | 1 | identity | 0.0005 | 0.0003 | 1.8067x | 0.9935x |
| util.img_as_bool | bool | stepped | 1 | identity | 0.0005 | 0.0003 | 1.8446x | 1.0086x |
| util.img_as_bool | uint8 | contiguous | 1 | native | 0.0496 | 0.0488 | 1.0176x | 0.9788x |
| util.img_as_bool | uint8 | fortran | 1 | native | 0.0598 | 0.0512 | 1.1684x | 13.3320x |
| util.img_as_bool | uint8 | reversed | 1 | native | 0.5505 | 0.4415 | 1.2470x | 0.9926x |
| util.img_as_bool | uint8 | stepped | 1 | native | 0.1253 | 0.1159 | 1.0815x | 1.1076x |
| util.img_as_bool | int8 | contiguous | 1 | native | 0.0664 | 0.0681 | 0.9750x | 0.7197x |
| util.img_as_bool | int8 | fortran | 1 | native | 0.0663 | 0.0671 | 0.9877x | 11.7698x |
| util.img_as_bool | int8 | reversed | 1 | native | 0.5162 | 0.3212 | 1.6070x | 0.8754x |
| util.img_as_bool | int8 | stepped | 1 | native | 0.0785 | 0.0807 | 0.9725x | 0.9543x |
| util.img_as_bool | uint16 | contiguous | 1 | native | 0.0692 | 0.0702 | 0.9862x | 0.9146x |
| util.img_as_bool | uint16 | fortran | 1 | native | 0.0721 | 0.0679 | 1.0617x | 39.3908x |
| util.img_as_bool | uint16 | reversed | 1 | native | 0.4145 | 0.3068 | 1.3510x | 1.0900x |
| util.img_as_bool | uint16 | stepped | 1 | native | 0.0868 | 0.0877 | 0.9901x | 1.0292x |
| util.img_as_bool | int16 | contiguous | 1 | native | 0.0761 | 0.0698 | 1.0910x | 0.8753x |
| util.img_as_bool | int16 | fortran | 1 | native | 0.0680 | 0.0705 | 0.9651x | 39.4174x |
| util.img_as_bool | int16 | reversed | 1 | native | 0.5945 | 0.4211 | 1.4118x | 0.9196x |
| util.img_as_bool | int16 | stepped | 1 | native | 0.0933 | 0.0919 | 1.0154x | 1.0254x |
| util.img_as_bool | uint32 | contiguous | 1 | native | 0.1202 | 0.1110 | 1.0823x | 0.9860x |
| util.img_as_bool | uint32 | fortran | 1 | native | 0.1158 | 0.1101 | 1.0521x | 48.3039x |
| util.img_as_bool | uint32 | reversed | 1 | native | 0.4809 | 0.5975 | 0.8048x | 0.8321x |
| util.img_as_bool | uint32 | stepped | 1 | native | 0.1084 | 0.1712 | 0.6331x | 0.8414x |
| util.img_as_bool | int32 | contiguous | 1 | native | 0.1083 | 0.1035 | 1.0464x | 1.0451x |
| util.img_as_bool | int32 | fortran | 1 | native | 0.1207 | 0.1139 | 1.0605x | 40.6632x |
| util.img_as_bool | int32 | reversed | 1 | native | 0.5036 | 0.5871 | 0.8578x | 1.1443x |
| util.img_as_bool | int32 | stepped | 1 | native | 0.1076 | 0.1287 | 0.8358x | 1.0299x |
| util.img_as_bool | uint64 | contiguous | 1 | native | 0.4446 | 0.4503 | 0.9874x | 0.9426x |
| util.img_as_bool | uint64 | fortran | 1 | native | 0.3376 | 0.3582 | 0.9426x | 33.4515x |
| util.img_as_bool | uint64 | reversed | 1 | native | 0.5428 | 2.8707 | 0.1891x | 1.0527x |
| util.img_as_bool | uint64 | stepped | 1 | native | 0.1976 | 0.2250 | 0.8782x | 1.0747x |
| util.img_as_bool | int64 | contiguous | 1 | native | 0.3447 | 0.3472 | 0.9928x | 0.9185x |
| util.img_as_bool | int64 | fortran | 1 | native | 0.3183 | 0.2710 | 1.1742x | 43.2756x |
| util.img_as_bool | int64 | reversed | 1 | native | 0.6443 | 3.9546 | 0.1629x | 0.9001x |
| util.img_as_bool | int64 | stepped | 1 | native | 0.2536 | 0.4714 | 0.5381x | 0.8143x |
| util.img_as_bool | float32 | contiguous | 1 | native | 0.1259 | 0.1211 | 1.0389x | 0.9831x |
| util.img_as_bool | float32 | fortran | 1 | native | 0.1192 | 0.1212 | 0.9832x | 45.0464x |
| util.img_as_bool | float32 | reversed | 1 | native | 0.4698 | 0.5425 | 0.8661x | 0.8790x |
| util.img_as_bool | float32 | stepped | 1 | native | 0.1206 | 0.1202 | 1.0034x | 0.9985x |
| util.img_as_bool | float64 | contiguous | 1 | native | 0.2104 | 0.2177 | 0.9665x | 1.2440x |
| util.img_as_bool | float64 | fortran | 1 | native | 0.2446 | 0.2929 | 0.8352x | 43.5822x |
| util.img_as_bool | float64 | reversed | 1 | native | 0.6054 | 3.7288 | 0.1624x | 0.8486x |
| util.img_as_bool | float64 | stepped | 1 | native | 0.1864 | 0.2422 | 0.7696x | 0.9510x |
| util.img_as_int | uint16 | full-contiguous | 1 | native | 0.1372 | 0.1883 | 0.7284x | 1.1176x |
| util.img_as_int | uint16 | full-fortran | 1 | native | 2.6732 | 3.5477 | 0.7535x | 0.9595x |
| util.img_as_int | uint16 | full-reversed | 1 | native | 2.4779 | 1.5162 | 1.6343x | 0.9105x |
| util.img_as_int | uint16 | full-stepped | 1 | native | 0.1857 | 0.2855 | 0.6506x | 1.0560x |
| util.img_as_int | uint16 | narrow-contiguous | 1 | native | 0.2067 | 0.2232 | 0.9264x | 0.8985x |
| util.img_as_int | uint16 | narrow-fortran | 1 | native | 0.5600 | 0.6730 | 0.8321x | 6.6818x |
| util.img_as_int | uint16 | narrow-reversed | 1 | native | 1.5815 | 2.2986 | 0.6880x | 0.8776x |
| util.img_as_int | uint16 | narrow-stepped | 1 | native | 0.2723 | 0.1861 | 1.4632x | 1.1094x |
