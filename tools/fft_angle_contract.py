"""Independent exact reconstruction of the existing FFT angle expression.
No candidate FFT arithmetic or scientific runtime is executed here.
"""
import argparse
from decimal import Decimal, localcontext, ROUND_HALF_EVEN, ROUND_FLOOR
from fractions import Fraction
import json
import struct
from pathlib import Path

PI_LITERAL = '3.141592653589793238462643383279502884197'
DENOMINATORS = tuple(sorted(set(range(1, 65)) | {67, 81, 97, 117, 125, 127, 128,
    234, 256, 257, 468, 509, 512, 1021, 1024, 2048, 4093, 4096, 8191}))


def round_fraction(value, precision):
    """Positive normal binary rounding, nearest with ties to even."""
    if value <= 0 or precision not in (53, 64): raise ValueError('positive value and assessed precision required')
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent: exponent -= 1
    scale = precision - 1 - exponent
    scaled = value * Fraction(2) ** scale
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (2 * remainder == scaled.denominator and whole % 2): whole += 1
    return Fraction(whole) * Fraction(2) ** -scale


def round_decimal(value, precision):
    """Separate Decimal engine; no Fraction or binary floating calculation."""
    exponent = int((value.ln() / Decimal(2).ln()).to_integral_value(rounding=ROUND_FLOOR))
    scale = precision - 1 - exponent
    return (value * Decimal(2) ** scale).to_integral_value(rounding=ROUND_HALF_EVEN) * Decimal(2) ** -scale


def angle_fraction(denominator, precision):
    if type(denominator) is not int or denominator < 1: raise ValueError('positive integer denominator required')
    pi = round_fraction(Fraction(PI_LITERAL), precision)
    quarter = round_fraction(pi / 4, precision)
    quotient = round_fraction(quarter / denominator, precision)
    return round_fraction(quotient, 53)


def angle_decimal(denominator, precision):
    with localcontext() as context:
        context.prec = 100
        pi = round_decimal(Decimal(PI_LITERAL), precision)
        quarter = round_decimal(pi / 4, precision)
        quotient = round_decimal(quarter / denominator, precision)
        return round_decimal(quotient, 53)


def contract():
    rows = []
    for n in DENOMINATORS:
        bits = {}
        for name, precision in (('x87_64_significand', 64), ('msvc_53_significand', 53)):
            exact = angle_fraction(n, precision)
            if exact != Fraction(angle_decimal(n, precision)): raise AssertionError('independent arithmetic engines disagree')
            bits[name] = struct.pack('>d', float(exact)).hex()
        rows.append(dict(denominator=n, **bits, same_final_binary64=len(set(bits.values())) == 1))
    return dict(schema='fft-angle-precision-v1', scope='Existing expression baseline, not a math candidate or LPI tolerance/oracle update',
        source_expression='double(0.25L*pi/denominator)', pi_decimal_literal=PI_LITERAL,
        rounding='nearest ties to even at literal, quarter multiply, division; then binary64',
        x87_significand_bits=64, msvc_significand_bits=53, independent_decimal_precision=100,
        denominators=list(DENOMINATORS), cases=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv); result = contract()
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    else: print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
