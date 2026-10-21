"""Conditioned inverse-Luv parity checks shared by tests and benchmarks."""
import numpy as np
from skimage import color as reference
import skimagexx as xx


def check_routed_hsv(image,fromspace,actual,expected):
    """Validate each stage and propagate matrix roundoff through HSV ratios."""
    from skimage.color import colorconv
    names={'rgb cie':'rgbcie2rgb','xyz':'xyz2rgb','yuv':'yuv2rgb','yiq':'yiq2rgb',
           'ypbpr':'ypbpr2rgb','ycbcr':'ycbcr2rgb','ydbdr':'ydbdr2rgb','hsv':'hsv2rgb'}
    space=fromspace.lower()
    rgb=getattr(xx.color,names[space])(image)
    reference_rgb=getattr(reference,names[space])(image)
    single=rgb.dtype==np.float32
    rtol=atol=2e-6 if single else 2e-13
    roundoff=0
    if single and space not in ('hsv','xyz'):
        matrix=getattr(colorconv,'rgb_from_'+space.replace(' ',''))
        roundoff=4*np.finfo(np.float32).eps*np.max(np.sum(np.abs(matrix),axis=1))*np.max(np.abs(image.astype(np.float64)))
    assert np.all(np.abs(rgb.astype(np.float64)-reference_rgb)<=atol+roundoff+rtol*np.abs(reference_rgb))
    np.testing.assert_allclose(actual,reference.rgb2hsv(rgb),rtol=rtol,atol=atol)
    # max/min are 1-Lipschitz, hence delta changes by at most 2*e.
    e=np.max(np.abs(rgb.astype(np.float64)-reference_rgb),axis=-1)
    m=reference_rgb.max(axis=-1).astype(np.float64)
    delta=m-reference_rgb.min(axis=-1)
    denominator=np.abs(m)-e
    with np.errstate(divide='ignore',invalid='ignore'):
        saturation_bound=np.where(denominator>0,(2+np.abs(expected[...,1]))*e/denominator,np.inf)
        hue_bound=np.where(delta>2*e,4*e/(delta-2*e),np.inf)
    bound=np.stack((hue_bound,saturation_bound,e),axis=-1)
    difference=np.abs(actual.astype(np.float64)-expected)
    # Hue is periodic; the interval may cross its 0/1 wrap.
    difference[...,0]=np.minimum(difference[...,0],1-difference[...,0])
    assert np.all(difference<=bound+atol+rtol*np.abs(expected))
    return {'parity_contract':'RGB stage tolerance and HSV ratio/wrapped-hue propagated bound',
            'route_rgb_max_abs_error':float(np.max(e))}


def check_float32_luv_inverse_stages(luv,actual=None,expected=None):
    # NumPy's dispatched float32 power and the platform powf can differ by
    # one ULP. Cancellation in Luv->XYZ followed by the near-black sRGB slope
    # amplifies that difference. Check the stages rather than relaxing the
    # RGB tolerance for every pixel or every other color conversion.
    expected_xyz=reference.luv2xyz(luv)
    xyz=xx.color.luv2xyz(luv)
    assert xyz.dtype==expected_xyz.dtype==np.float32
    np.testing.assert_array_max_ulp(xyz[...,1],expected_xyz[...,1],maxulp=1)
    np.testing.assert_allclose(xyz,expected_xyz,rtol=2e-6,atol=2e-6)

    white=reference.xyz_tristimulus_values(illuminant='D65',observer='2',dtype=np.float32)
    denominator=np.array([1,15,3],dtype=np.float32)@white
    light,u,v=np.moveaxis(luv,-1,0)
    y=xyz[...,1]
    a=4*white[0]/denominator+u/(13*light+np.finfo(np.float32).eps)
    b=9*white[1]/denominator+v/(13*light+np.finfo(np.float32).eps)
    c=3*y*(5*b-3)
    z=((a-4)*c-15*a*b*y)/(12*b)
    x=-(c/b+3*z)
    reconstructed=np.stack((x,y,z),axis=-1)
    # All remaining XYZ arithmetic must agree once the power result agrees.
    np.testing.assert_array_equal(xyz,reconstructed)
    if actual is None:
        actual=xx.color.luv2rgb(luv)
    if expected is None:
        expected=reference.luv2rgb(luv)
    assert actual.dtype==expected.dtype==np.float32
    np.testing.assert_allclose(actual,reference.xyz2rgb(reconstructed),rtol=2e-6,atol=2e-6)

    # Also bound the direct comparison against the original implementation.
    # The clipped sRGB transfer has maximum derivative 12.92; multiplying the
    # measured XYZ perturbation by |M| propagates an absolute error bound.
    from skimage.color.colorconv import rgb_from_xyz
    bound=12.92*(np.abs(xyz.astype(np.float64)-expected_xyz)@np.abs(rgb_from_xyz.T))
    assert np.all(np.abs(actual.astype(np.float64)-expected)<=bound+2e-6*(1+np.abs(expected)))

    return {'parity_contract':'float32 Luv: one-ULP Y, exact remaining XYZ, base RGB stage and propagated direct bound',
            'xyz_max_abs_error':float(np.max(np.abs(xyz-expected_xyz))),
            'rgb_max_propagated_bound':float(np.max(bound))}
