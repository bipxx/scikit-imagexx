"""Exact retained original output and input reconstruction for Hough ellipses."""

from hashlib import sha256
import json
from pathlib import Path


REFERENCE = Path(__file__).resolve().parents[1] / 'coverage/contracts/hough-ellipse-reference-v1.jsonl'
REFERENCE_SHA256 = '0d2ba0b822bf4354c2f37f02a84e16bf59fadc1ea9b3db5050d85accf6f47e49'


def frozen_rows():
    content = REFERENCE.read_bytes()
    assert sha256(content.replace(b'\r\n', b'\n')).hexdigest() == REFERENCE_SHA256
    return tuple(json.loads(row) for row in content.splitlines())


def snapshot(value, content=False):
    data = value.tobytes()
    result = dict(type=type(value).__name__, dtype=value.dtype.str, metadata=value.dtype.metadata is not None,
                  size=value.size, shape=list(value.shape), strides=list(value.strides),
                  c_order=value.flags.c_contiguous, f_order=value.flags.f_contiguous,
                  aligned=value.flags.aligned, owns=value.flags.owndata, writable=value.flags.writeable,
                  base=type(value.base).__name__, sha256=sha256(data).hexdigest())
    if content:
        result['data_hex'] = data.hex()
    return result


def restore(np, row):
    record = row['image']
    image = np.empty(record['shape'], dtype=record['dtype'])
    image[...] = np.frombuffer(bytes.fromhex(record['data_hex']), dtype=image.dtype).reshape(image.shape)
    if record['strides'][-1] < 0:
        image = image[:, ::-1].copy(order='C')[:, ::-1]
    assert snapshot(image, content=True) == record
    return image


def result_record(np, value):
    record = snapshot(value, value.size <= 64)
    record['dtype_description'] = [list(field) for field in value.dtype.descr]
    record['itemsize'] = value.itemsize
    if value.dtype.names:
        record['field_offsets'] = {name: value.dtype.fields[name][1] for name in value.dtype.names}
        record['fields_sha256'] = {name: sha256(value[name].tobytes()).hexdigest() for name in value.dtype.names}
        record['first_rows'] = [list(row) for row in value[:4].tolist()]
        record['last_rows'] = [list(row) for row in value[-2:].tolist()]
        record['maximum_votes'] = int(np.max(value['accumulator'])) if value.size else None
        record['orientation_range'] = (
            [float(np.min(value['orientation'])), float(np.max(value['orientation']))] if value.size else None
        )
    return record
