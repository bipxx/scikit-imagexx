"""Frozen default-only normalized/Hu error policy, independent of tests.

Pass the actual first algorithm argument (mu/nu), not an image from which a
caller has prepared moments. Default normalization is order3/spacingNone.
Operation counts and envelopes match the pre-output prospective policy.
"""
import itertools
import numpy as np

_HU_OPERATIONS=(1,5,7,3,16,11,16)


def derived_default_limits(api,image,expected):
    if api not in ("measure.moments_normalized","measure.moments_hu"):
        raise ValueError("unknown derived-moments API")
    if expected.dtype.kind not in "fc":return np.zeros(expected.shape,dtype=np.float64)
    eps=float(np.finfo(expected.real.dtype).eps);u=eps/2
    if api=="measure.moments_normalized":
        n=160 if expected.dtype.kind=="c"else 24
        bound=(n*u/(1-n*u))*np.abs(expected.astype(np.complex128 if expected.dtype.kind=="c"else np.float64))
    else:
        values=image.real.astype(np.float32 if image.dtype==np.dtype("float32")else np.float64)
        a,b=abs(float(values[3,0]))+abs(float(values[1,2])),abs(float(values[2,1]))+abs(float(values[0,3]))
        q,r=a*a,b*b;n4=4*abs(float(values[1,1]));s=abs(float(values[2,0]))+abs(float(values[0,2]))
        t0,t1=a*(q+3*r),b*(3*q+r);c,d=abs(float(values[3,0]))+3*abs(float(values[1,2])),3*abs(float(values[2,1]))+abs(float(values[0,3]))
        envelopes=(s,s*s+n4*abs(float(values[1,1])),c*c+d*d,q+r,c*t0+d*t1,s*(q+r)+n4*a*b,d*t0+c*t1)
        bound=np.asarray([(2*n+2)*u/(1-(2*n+2)*u)*scale for n,scale in zip(_HU_OPERATIONS,envelopes)])
    bound=np.asarray(bound,dtype=np.float64);bound[~np.isfinite(expected)]=0
    if not np.all(np.isfinite(bound[np.isfinite(expected)]))or not np.all(bound>=0):
        raise AssertionError("derived moments reference has an invalid allowance")
    return bound


def derived_default_error(api,image,expected,actual):
    expected,actual=np.asarray(expected),np.asarray(actual)
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.dtype.metadata!=expected.dtype.metadata:
        raise AssertionError("derived moments output shape or dtype differs")
    with np.errstate(all="ignore"):bound=derived_default_limits(api,image,expected)
    if expected.dtype.kind not in "fc":
        np.testing.assert_array_equal(actual,expected);return 0.
    components=((actual.real,expected.real),(actual.imag,expected.imag))if expected.dtype.kind=="c"else((actual,expected),)
    for a,e in components:
        for mask in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(a),mask(e))
        finite=np.isfinite(e)
        if not np.all(np.isfinite(bound[finite])&(bound[finite]>=0)):
            raise AssertionError("derived moments reference has an invalid component allowance")
        if not np.all(np.abs(a[finite]-e[finite])<=bound[finite]):
            raise AssertionError("derived moments values exceed the frozen cell-specific allowance")
        exact=(bound==0)&finite
        np.testing.assert_array_equal(a[exact],e[exact]);np.testing.assert_array_equal(np.signbit(a[exact]),np.signbit(e[exact]))
    if api=="measure.moments_normalized":
        if expected.shape!=image.shape:raise AssertionError("normalized moments changed the input shape")
        visited=np.zeros(expected.shape,dtype=bool)
        for powers in itertools.product(range(4),repeat=expected.ndim):
            visited[powers]=True
            if sum(powers)<2 and not(np.isnan(actual[powers])and np.isnan(expected[powers])):
                raise AssertionError("normalized low-degree NaN differs")
        for a,e in components:
            np.testing.assert_array_equal(a[~visited],e[~visited])
            np.testing.assert_array_equal(np.signbit(a[~visited]),np.signbit(e[~visited]))
    elif expected.shape!=(7,):raise AssertionError("Hu moments must have seven outputs")
    finite=np.isfinite(expected)
    return float(np.max(np.abs(actual[finite]-expected[finite]),initial=0.))
