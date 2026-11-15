"""Replay captured FP INPUT words on a trusted current-thread FEnv32 template.

Instruction/data addresses and reserved bytes are diagnostics, not portable
numerical input. No target/observer AFTER word is changed by this fixture.
"""
FIELDS=((0,2),(4,2),(28,4))


def compatible(raw):
    assert type(raw)is bytes and len(raw)==32,'Expected retained glibc32 envelope'
    control=int.from_bytes(raw[0:2],'little');tags=int.from_bytes(raw[8:10],'little');mxcsr=int.from_bytes(raw[28:32],'little')
    assert tags==0xffff,'Saved x87 data stack must be empty'
    assert control&0x3f==0x3f and mxcsr&0x1f80==0x1f80,'Only masked exceptions supported'
    assert mxcsr&0xffff0000==0,'Reserved MXCSR bits forbidden'


def recorded_input(current,recorded_hex):
    recorded=bytes.fromhex(recorded_hex);compatible(current);compatible(recorded)
    out=bytearray(current)
    for offset,width in FIELDS:out[offset:offset+width]=recorded[offset:offset+width]
    return bytes(out)
