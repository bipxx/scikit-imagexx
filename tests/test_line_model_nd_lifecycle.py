"""Original-first LineModelND lifecycle. See third_party/scikit-image-LICENSE.txt."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np

from skimagexx import measure
from skimage._shared.utils import FailedEstimation


def digest(raw):
    return sha256(raw).hexdigest()


def run_rows(lm, failure_type):
    def view(a):
        base = a.base
        return dict(type=type(a).__name__,dtype=a.dtype.str,metadata=None if a.dtype.metadata is None else dict(a.dtype.metadata),
                    shape=list(a.shape),strides=list(a.strides),owns=a.flags.owndata,c=a.flags.c_contiguous,
                    f=a.flags.f_contiguous,align=a.flags.aligned,write=a.flags.writeable,
                    base=None if base is None else dict(type=type(base).__name__,shape=list(base.shape),
                        strides=list(base.strides),owns=base.flags.owndata,dtype=base.dtype.str),
                    bytes=a.tobytes().hex() if a.size <= 32 else None,sha256=digest(a.tobytes()))

    def plain(value):
        if isinstance(value,np.ndarray):
            return view(value)
        if isinstance(value,lm):
            return dict(type=type(value).__name__,state={key:plain(getattr(value,key)) for key in ("origin","direction")})
        if isinstance(value,failure_type):
            try:
                value.origin
            except AttributeError as error:
                access = [type(error).__name__, str(error)]
            else:
                raise AssertionError("failure allowed origin")
            try:
                value()
            except AttributeError as error:
                call_error = [type(error).__name__,str(error)]
            else:
                raise AssertionError("failure was callable")
            return dict(type=type(value).__name__,truth=bool(value),message=value.message,representation=repr(value),
                        canonical_type=type(value) is failure_type,access=access,call=call_error)
        if isinstance(value,tuple):
            return [plain(item) for item in value]
        return value

    rows = {}

    def probe(name, action, inputs=()):
        before = [view(item) for item in inputs]
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                result = action()
            except Exception as error:
                result, status = None, {"error":type(error).__name__,"message":str(error)}
            else:
                status = {"result":plain(result)}
        assert [view(item) for item in inputs] == before
        rows[name] = dict(status,warnings=[[notice.category.__name__,str(notice.message)] for notice in seen],
                          inputs=before)
        return result

    origin = np.array([1.,2.,3.])
    direction = np.array([2.,-1.,0.5])
    m = probe("constructor-copies",lambda:lm(origin,direction),(origin,direction))
    rows["constructor-copies"]["aliases"] = [np.shares_memory(origin,m.origin),
                                              np.shares_memory(direction,m.direction)]
    pair = probe("params-identity",lambda:m.params)
    rows["params-identity"]["same_members"] = [pair[0] is m.origin,pair[1] is m.direction]
    pair[0][1] = 19.
    rows["params-identity"]["mutation_visible"] = bool(m.origin[1] == 19.)
    rows["params-identity"]["input_untouched"] = bool(origin[1] == 2.)
    m.origin[1] = 2.
    probe("constructor-lists",lambda:lm([1,2],[3,4]))
    probe("constructor-mismatch",lambda:lm([1,2],[3]))
    probe("constructor-scalar",lambda:lm(1,2))
    blank = probe("blank",lm)
    probe("blank-predict",lambda:blank.predict(np.array([2.])))
    probe("blank-residuals",lambda:blank.residuals(np.ones((2,2))))
    probe("constructor-pickle-roundtrip",lambda:pickle.loads(pickle.dumps(m)))
    two = np.array([[1.,2.,3.],[4.,6.,8.]])
    one = two[:1]
    fail = probe("from-estimate-failure",lambda:lm.from_estimate(one),(one,))
    assert isinstance(fail,failure_type)
    probe("from-estimate-empty",lambda:lm.from_estimate(two[:0]))
    probe("from-estimate-two",lambda:lm.from_estimate(two),(two,))
    rev = two[::-1]
    probe("from-estimate-reversed",lambda:lm.from_estimate(rev),(rev,))
    probe("from-estimate-f32",lambda:lm.from_estimate(two.astype(np.float32)))
    probe("from-estimate-i16",lambda:lm.from_estimate(two.astype(np.int16)))
    tagged = two.view(np.dtype(np.float64,metadata={"tag":"line"}))
    probe("from-estimate-tagged",lambda:lm.from_estimate(tagged),(tagged,))
    probe("from-estimate-zero",lambda:lm.from_estimate(np.ones((2,3))))
    probe("from-estimate-bad-rank",lambda:lm.from_estimate(np.array([1.,2.])))
    probe("from-estimate-bad-width",lambda:lm.from_estimate(np.ones((3,1))))
    i = np.arange(128)
    t = i / 4 - 16
    many = np.column_stack((t,2*t+(i%3-1)/8,-t+(i%7-3)/16))
    probe("from-estimate-svd",lambda:lm.from_estimate(many),(many,))
    probe("from-estimate-tied-svd",lambda:lm.from_estimate(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])))
    probe("from-estimate-svd-invalid",lambda:lm.from_estimate(np.full((3,2),np.nan)))

    class Subclass(lm):
        pass
    probe("from-estimate-subclass",lambda:Subclass.from_estimate(two))
    old = (m.origin,m.direction)
    probe("estimate-failure-keeps-state",lambda:m.estimate(one),(one,))
    rows["estimate-failure-keeps-state"]["identical_state_objects"] = [m.origin is old[0],m.direction is old[1]]
    rows["estimate-failure-keeps-state"]["state"] = plain(m)
    probe("estimate-error-keeps-state",lambda:m.estimate(np.ones(3)))
    rows["estimate-error-keeps-state"]["identical_state_objects"] = [m.origin is old[0],m.direction is old[1]]
    probe("estimate-success-replaces-state",lambda:m.estimate(two),(two,))
    rows["estimate-success-replaces-state"]["identical_state_objects"] = [m.origin is old[0],m.direction is old[1]]
    rows["estimate-success-replaces-state"]["state"] = plain(m)
    fitted = (m.origin,m.direction)
    probe("estimate-repeated-svd",lambda:m.estimate(many),(many,))
    rows["estimate-repeated-svd"]["identical_state_objects"] = [m.origin is fitted[0],m.direction is fitted[1]]
    rows["estimate-repeated-svd"]["state"] = plain(m)
    probe("estimate-empty-keeps-state",lambda:m.estimate(two[:0]))
    rows["estimate-empty-keeps-state"]["state"] = plain(m)

    model = lm([1.,2.,3.],[2.,-1.,0.5])
    points = np.array([[2.,3.,4.],[-3.,5.,7.],[5.,0.,-1.]])
    x = np.array([0.125,-2.,0.,4.5])
    custom = ([2.,4.,6.],[1.,2.,3.])
    matrix = x.reshape(2,2)
    controls = (
        ("residuals-nonunit",lambda:model.residuals(points),(points,)),
        ("residuals-strided",lambda:model.residuals(points[::-1]),()),
        ("residuals-empty",lambda:model.residuals(points[:0]),()),
        ("residuals-bad-rank",lambda:model.residuals(x, params=(1,)),(x,)),
        ("residuals-origin-dim",lambda:model.residuals(points[:,:2]),()),
        ("residuals-deprecated",lambda:model.residuals(points,params=custom),(points,)),
        ("residuals-explicit-none",lambda:model.residuals(points,params=None),(points,)),
        ("predict-default",lambda:model.predict(x),(x,)),
        ("predict-neg-axis",lambda:model.predict(x,axis=-1),(x,)),
        ("predict-bad-axis",lambda:model.predict(x,axis=3),(x,)),
        ("predict-noninteger-axis",lambda:model.predict(x,axis=0.5),(x,)),
        ("predict-zero-direction",lambda:lm([1,2],[0,1]).predict(x),(x,)),
        ("predict-scalar",lambda:model.predict(4.),()),
        ("predict-matrix",lambda:model.predict(matrix),(matrix,)),
        ("predict-empty",lambda:model.predict(x[:0]),()),
        ("predict-list",lambda:model.predict([1,2]),()),
        ("predict-invalid-params",lambda:model.predict(x,params=(1,)),(x,)),
        ("predict-deprecated",lambda:model.predict(x,params=custom),(x,)),
        ("predict-x",lambda:model.predict_x(x),(x,)),
        ("predict-y",lambda:model.predict_y(x),(x,)),
        ("predict-x-matrix",lambda:model.predict_x(matrix),(matrix,)),
        ("predict-y-scalar",lambda:model.predict_y(2.),()),
        ("predict-x-deprecated",lambda:model.predict_x(x,params=custom),(x,)),
        ("predict-y-explicit-none",lambda:model.predict_y(x,params=None),(x,)),
    )
    for name, action, operands in controls:
        probe(name,action,operands)
    return rows


def test_frozen_lifecycle():
    path = Path(__file__).parent / "fixtures/line_model_nd_026.json"
    expected = json.loads(path.read_bytes())
    rows = expected["common_rows"]
    if sys.platform.startswith("linux") and np.__version__ == "1.24.4":
        rows.update(expected["minimum_delta"])
    elif not (sys.platform == "win32" and np.__version__ == "2.5.4"):
        from skimage.measure import LineModelND as StockLineModelND
        rows = run_rows(StockLineModelND, FailedEstimation)
    api = measure.LineModelND
    assert str(inspect.signature(api)) == expected["class_signature"]
    for name, desc in expected["members"].items():
        member = getattr(api, name)
        assert type(inspect.getattr_static(api, name)).__name__ == desc["descriptor"]
        if desc["signature"] is not None:
            assert str(inspect.signature(member)) == desc["signature"]
    measured = run_rows(api, FailedEstimation)
    assert measured == rows


def test_prediction_aliases_keep_dynamic_dispatch_and_owner():
    produced = np.arange(12.0).reshape(4, 3)
    x = np.arange(4.0)
    calls = []

    class Model(measure.LineModelND):
        def predict(self, values, axis=0):
            calls.append((self, values, axis))
            return produced

    model = Model([1., 2., 3.], [2., 3., 4.])
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        alias_x = model.predict_x(x)
        alias_y = model.predict_y(x, params=None)
        explicit = model.predict_x(x, params=([10., 20., 30.], [4., 5., 6.]))
    assert calls[0][0] is calls[1][0] is model
    assert all(values is x for _, values, _ in calls)
    assert [axis for _, _, axis in calls] == [1, 0, 1]
    new_model = calls[2][0]
    assert type(new_model) is Model and new_model is not model
    np.testing.assert_array_equal(new_model.origin, [10., 20., 30.])
    assert all(result.base is produced.base for result in (alias_x, alias_y, explicit))
    np.testing.assert_array_equal(alias_x, produced[:, 0])
    np.testing.assert_array_equal(alias_y, produced[:, 1])
    np.testing.assert_array_equal(explicit, produced[:, 0])
    assert [(notice.category, str(notice.message).split("documentation of ")[-1])
            for notice in notices] == [(FutureWarning, "`LineModelND.predict_y`."),
                                       (FutureWarning, "`LineModelND.predict_x`.")]

    def instance_predict(values, axis):
        assert values is x and axis == 0
        return produced

    model.predict = instance_predict
    assert model.predict_y(x).base is produced.base


def test_large_two_coordinate_prediction():
    model = measure.LineModelND([1., 2.], [3., 4.])
    x = np.arange(8192) / 8 - 128
    original = x.tobytes()

    def expected(values, axis, origin=model.origin, direction=model.direction):
        length = (values - origin[axis]) / direction[axis]
        return origin + length[..., np.newaxis] * direction

    for axis in (0, 1, -2, -1):
        result = model.predict(x, axis)
        exact = expected(x, axis)
        assert result.tobytes() == exact.tobytes()
        assert result.dtype == exact.dtype and result.shape == exact.shape and result.strides == exact.strides
        assert result.flags.owndata and result.flags.writeable and result.base is None
    for call, axis, col in ((model.predict_x, 1, 0), (model.predict_y, 0, 1)):
        result = call(x)
        exact = expected(x, axis)
        assert result.tobytes() == exact[:, col].tobytes()
        assert not result.flags.owndata and result.flags.writeable
        assert result.strides == exact[:, col].strides
        assert result.base.flags.owndata and result.base.base is None
        assert result.base.tobytes() == exact.tobytes()
    assert x.tobytes() == original

    extreme = measure.LineModelND([2.**-128, -2.**128], [2.**128, 2.**-128])
    edge = np.array([0., -0., 2.**-128, np.nextafter(2.**-128, np.inf), 2.**128] * 32)
    assert extreme.predict(edge, axis=1).tobytes() == expected(
        edge, 1, extreme.origin, extreme.direction).tobytes()

    tagged = x.view(np.dtype(np.float64, metadata={"tag": "line"}))
    alias_f64 = x.astype(np.dtype("float64").newbyteorder(">").newbyteorder("="))
    for alternate in (x[::2], tagged, alias_f64, x.astype(np.float32)):
        result = model.predict(alternate)
        exact = expected(alternate, 0)
        assert result.tobytes() == exact.tobytes() and result.dtype.metadata == exact.dtype.metadata
    for value in (np.nan, np.inf, np.finfo(np.float64).smallest_subnormal):
        unsupported = np.full(128, value)
        assert model.predict(unsupported).tobytes() == expected(unsupported, 0).tobytes()
    zero_axis = measure.LineModelND([1., 2.], [0., 3.])
    try:
        zero_axis.predict(x)
    except ValueError as error:
        assert str(error) == "Line parallel to axis 0"
    else:
        raise AssertionError("zero direction was accepted")
    huge = np.full(128, np.finfo(np.float64).max)
    with np.errstate(over="raise"):
        small_axis = measure.LineModelND([0., 2.], [2.**-128, 1.])
        try:
            small_axis.predict(huge)
        except FloatingPointError as error:
            assert "overflow encountered in divide" in str(error)
        else:
            raise AssertionError("overflow bypassed the active NumPy error policy")


def test_large_three_coordinate_prediction():
    model = measure.LineModelND([1., 2., 3.], [2., -1., 0.5])
    x = np.arange(8192) / 8 - 128
    before = x.tobytes()

    def expected(values, axis):
        length = (values - model.origin[axis]) / model.direction[axis]
        return model.origin + length[..., np.newaxis] * model.direction

    for axis in (0, 1, 2, -3, -2, -1):
        result = model.predict(x, axis)
        assert result.tobytes() == expected(x, axis).tobytes()
        assert result.dtype == np.float64 and result.shape == (8192, 3)
        assert result.strides == (24, 8) and result.base is None
        assert result.flags.owndata and result.flags.writeable and result.flags.c_contiguous
    for call, axis, col in ((model.predict_x, 1, 0), (model.predict_y, 0, 1)):
        result = call(x)
        owner = expected(x, axis)
        assert result.tobytes() == owner[:, col].tobytes()
        assert result.strides == (24,) and not result.flags.owndata and result.flags.writeable
        assert result.base.tobytes() == owner.tobytes() and result.base.shape == (8192, 3)
        assert result.base.flags.owndata and result.base.base is None
    assert x.tobytes() == before

    model.origin = np.array([-0., 0., 0.])
    model.direction = np.array([2., -0., -1.])
    zeroes = np.array([0., -0., 2.**-128, -2.**128] * 32)
    assert model.predict(zeroes, axis=0).tobytes() == expected(zeroes, 0).tobytes()
    for alternative in (x[::2], x.astype(np.float32),
                        x.view(np.dtype(np.float64, metadata={"tag": "line"}))):
        assert model.predict(alternative, axis=2).tobytes() == expected(alternative, 2).tobytes()
    assert model.predict(x, axis=np.int64(0)).tobytes() == expected(x, 0).tobytes()
    try:
        model.predict(x, axis=1)
    except ValueError as error:
        assert str(error) == "Line parallel to axis 1"
    else:
        raise AssertionError("negative zero direction was accepted")
    try:
        model.predict(x, axis=3)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range prediction axis was accepted")
