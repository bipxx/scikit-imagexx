"""STDlib angle baseline guards; no scientific imports or FFT calls."""
import importlib.util,json,struct
from fractions import Fraction
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('angle_contract',ROOT/'tools/fft_angle_contract.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def test_fraction_and_independent_decimal_models_match_frozen_contract():
    frozen=json.loads((ROOT/'coverage/contracts/fft-angle-precision.json').read_text())
    assert a.contract()==frozen
    assert set(range(1,65)) <= set(frozen['denominators'])
    assert {9,13,18,26,36,52,117,234,468} <= set(frozen['denominators'])


@pytest.mark.parametrize('n,same,bits',[(9,True,'3fb657184ae74487'),(13,False,'3faeeebf2ca2ada8'),
    (18,True,'3fa657184ae74487'),(26,False,'3f9eeebf2ca2ada8'),(36,True,'3f9657184ae74487'),(52,False,'3f8eeebf2ca2ada8')])
def test_actual_target_geometry_final_bits(n,same,bits):
    row=next(r for r in a.contract()['cases'] if r['denominator']==n)
    assert row['same_final_binary64'] is same and row['x87_64_significand']==bits
    if not same: assert int(row['x87_64_significand'],16)-int(row['msvc_53_significand'],16)==1


@pytest.mark.parametrize('precision',(53,64))
def test_nearest_even_ties_are_resolved_without_host_float(precision):
    assert a.round_fraction(Fraction(1)+Fraction(1,2**precision),precision)==1
    assert a.round_fraction(Fraction(1)+Fraction(3,2**precision),precision)==Fraction(1)+Fraction(1,2**(precision-2))


@pytest.mark.parametrize('bad',(0,-1,True,1.5,'13'))
def test_denominator_boundary(bad):
    with pytest.raises(ValueError): a.angle_fraction(bad,64)


def test_freeze_refuses_to_overwrite_existing_baseline(tmp_path):
    target=tmp_path/'contract.json';target.write_text('immutable')
    with pytest.raises(FileExistsError):a.main(['--output',str(target)])
    assert target.read_text()=='immutable'


def test_source_literal_is_the_unchanged_upstream_baseline():
    import hashlib
    source=(ROOT/'third_party/pocketfft/pocketfft_hdronly.h').read_bytes()
    assert hashlib.sha256(source).hexdigest()=='50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c'
    assert ('constexpr auto pi = '+a.PI_LITERAL+'L;').encode() in source
