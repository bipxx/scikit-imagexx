"""Prospective derived-moment inputs and operation-derived allowances.

No scientific imports at collection. No changes to the earlier moments cohort.
Normalization has two scalar powers, two divisions and output storage: budget24
real rounding units (8 per pow,2 per division,4 for casts/promotions). Complex
power uses log/exp/trigonometric components:160 units cover two 6-component
power paths at8 units plus complex division/casts. These are declared bounded
libm budgets, not universal correctly-rounded guarantees. Hu uses the literal
polynomial's per-output path counts, doubled for two independently rounded
implementations plus2 conversions. Absolute term envelopes protect cancellation.
"""
import itertools

APIS = ("moments_normalized", "moments_hu")
STAGES = {"moments_normalized": "moments_normalize", "moments_hu": "moments_hu"}
HU_OPERATIONS = (1, 5, 7, 3, 16, 11, 16)
UNASSESSED = ("Hu rank2 with an axis smaller than4: no original call without installed bounds proof",
              "forged/stale ndarray storage", "arbitrary object/protocol arrays", "extended float dtypes")


def cases():
    rows = []
    def add(ident, api="moments_normalized", dtype="float64", shape=(5,6),
            order=3, spacing=None, layout="C", pattern="positive", policy=None, exact=False):
        rows.append(dict(id=ident,api=api,dtype=dtype,shape=shape,order=order,
                         spacing=spacing,layout=layout,pattern=pattern,policy=policy or {},exact=exact))
    for shape in ((5,), (5,6), (5,5,5)):
        for dtype in ("float16","float32","float64"):
            add(f"normalized-{len(shape)}d-{dtype}",dtype=dtype,shape=shape)
    for dtype in ("int8","uint8","int64","bool","complex64","complex128"):
        add("normalized-dtype-"+dtype,dtype=dtype)
    for order in (0,1,2,4,-1):add("normalized-order-"+str(order),order=order)
    for layout in ("F","step","reverse","endian","metadata","readonly"):
        add("normalized-layout-"+layout,dtype="float32",layout=layout)
    for ident,spacing in (("anisotropic",(.5,2.)),("array-f32",{"dtype":"float32","values":(.5,2.)}),
            ("array-f64",{"dtype":"float64","values":(.5,2.)}),("negative",(-.5,2.)),
            ("zero",(0.,1.)),("empty",()),("short",(.5,)),("complex",(1+.25j,2+.5j))):
        add("normalized-spacing-"+ident,spacing=spacing)
    for pattern in ("zero","negative-mass","negative-zero","nan","inf","small-mass","large-mass"):
        add("normalized-"+pattern,pattern=pattern)
    add("normalized-shape-short",shape=(3,4))
    add("normalized-scalar",shape=())
    add("normalized-empty",shape=(0,4))
    add("normalized-order-float",order=3.)
    add("normalized-exact-mass-one",dtype="float32",shape=(4,4),pattern="mass-one",exact=True)
    for dtype in ("float16","float32","float64","int32","bool","complex64","complex128"):
        add("hu-dtype-"+dtype,"moments_hu",dtype,shape=(4,4))
    for layout,dtype in (("F","float64"),("step","float64"),("reverse","float64"),
            ("endian","float32"),("metadata","float32"),("readonly","float64"),
            ("readonly","float16"),("unaligned","float64")):
        add("hu-layout-"+layout+"-"+dtype,"moments_hu",dtype,shape=(4,4),layout=layout)
    for pattern in ("points","reflected","rotated"):
        add("hu-"+pattern,"moments_hu",shape=(4,4),pattern=pattern)
    for pattern in ("zero","diagonal","near-cancel","nan","inf","overflow"):
        add("hu-"+pattern,"moments_hu",shape=(4,4),pattern=pattern,exact=pattern in ("zero","diagonal"))
    add("hu-wrong-rank","moments_hu",shape=(16,))
    add("hu-oversized","moments_hu",shape=(5,6))
    add("normalized-zero-raise",pattern="zero",policy={"all":"raise"})
    add("normalized-negative-mass-raise",pattern="negative-mass",policy={"all":"raise"})
    add("normalized-small-mass-warn",pattern="small-mass",policy={"all":"warn"})
    add("hu-overflow-raise","moments_hu",shape=(4,4),pattern="overflow",policy={"all":"raise"})
    assert len(rows)==76 and len({r["id"] for r in rows})==76
    return rows


def fixture(spec,np):
    shape=spec["shape"];dtype=np.dtype(spec["dtype"])
    data=(np.arange(int(np.prod(shape)),dtype=np.float64).reshape(shape)%7+1)
    if dtype.kind in "fc":data=data/16
    data=data.astype(dtype)
    if dtype.kind=="c":data.flat[0]=1+.5j
    pattern=spec["pattern"]
    if pattern=="zero":data[...] = 0
    elif pattern=="negative-mass":data.flat[0]=-2
    elif pattern=="negative-zero":data.flat[0]=-0.
    elif pattern in ("nan","inf"):data[...] = getattr(np,pattern)
    elif pattern=="small-mass":data.flat[0]=np.finfo(dtype).tiny
    elif pattern=="large-mass":data.flat[0]=np.finfo(dtype).max/4
    elif pattern=="mass-one":data.flat[0]=1
    elif pattern=="diagonal":data[...] = 0;data[2,0]=1;data[0,2]=2
    elif pattern=="near-cancel":
        data[...] = 0;data[3,0]=1;data[1,2]=np.nextafter(1.,2.);data[2,1]=1;data[0,3]=1
    elif pattern=="overflow":data[...] = np.finfo(dtype).max/2
    elif pattern in ("points","reflected","rotated"):
        points=[(-2,-1),(-2,1),(1,-1),(1,2)]
        if pattern=="reflected":points=[(x,-y)for x,y in points]
        if pattern=="rotated":points=[(-y,x)for x,y in points]
        cx=sum(x for x,y in points)/4;cy=sum(y for x,y in points)/4
        for i,j in itertools.product(range(4),repeat=2):
            data[i,j]=sum((x-cx)**i*(y-cy)**j for x,y in points)/(2.**(i+j+2))
    layout=spec["layout"]
    if layout=="F":data=np.asfortranarray(data)
    elif layout=="step":
        backing=np.zeros(tuple(2*n for n in shape),dtype=dtype);view=backing[tuple(slice(None,None,2)for n in shape)];view[...]=data;data=view
    elif layout=="reverse":data=data[tuple(slice(None,None,-1)for n in shape)]
    elif layout=="endian":data=data.astype(dtype.newbyteorder("S"))
    elif layout=="metadata":data=data.view(np.dtype(dtype,metadata={"derived-contract":True}))
    elif layout=="readonly":data.flags.writeable=False
    elif layout=="unaligned":
        backing=np.empty(data.nbytes+1,dtype=np.uint8);view=np.ndarray(shape,dtype=dtype,buffer=backing,offset=1);view[...]=data;data=view
    spacing=spec["spacing"]
    if isinstance(spacing,dict):spacing=np.array(spacing["values"],dtype=spacing["dtype"])
    options={}if spec["api"]=="moments_hu"else dict(order=spec["order"],spacing=spacing)
    return data,options,[]


def error_limits(spec,image,expected,np):
    if spec["exact"] or expected.dtype.kind not in "fc":return np.zeros(expected.shape,dtype=np.float64)
    eps=float(np.finfo(expected.real.dtype).eps);u=eps/2
    if spec["api"]=="moments_normalized":
        n=160 if expected.dtype.kind=="c"else 24
        bound=(n*u/(1-n*u))*np.abs(expected.astype(np.complex128 if expected.dtype.kind=="c"else np.float64))
    else:
        # Use the same source dtype conversion before the polynomial envelope.
        values=image.real.astype(np.float32 if image.dtype==np.dtype("float32") else np.float64)
        a,b=abs(float(values[3,0]))+abs(float(values[1,2])),abs(float(values[2,1]))+abs(float(values[0,3]))
        q,r=a*a,b*b;n4=4*abs(float(values[1,1]));s=abs(float(values[2,0]))+abs(float(values[0,2]))
        t0,t1=a*(q+3*r),b*(3*q+r);c,d=abs(float(values[3,0]))+3*abs(float(values[1,2])),3*abs(float(values[2,1]))+abs(float(values[0,3]))
        envelopes=(s,s*s+n4*abs(float(values[1,1])),c*c+d*d,q+r,c*t0+d*t1,s*(q+r)+n4*a*b,d*t0+c*t1)
        bound=np.asarray([(2*n+2)*u/(1-(2*n+2)*u)*scale for n,scale in zip(HU_OPERATIONS,envelopes)])
    bound=np.asarray(bound,dtype=np.float64);bound[~np.isfinite(expected)]=0
    assert np.all(np.isfinite(bound[np.isfinite(expected)]))
    return bound
