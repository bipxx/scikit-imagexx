"""Local file digests through Pooch or the standard library."""

try:
    from pooch import file_hash
except ModuleNotFoundError:
    import hashlib

    _sha256 = hashlib.sha256
    _sha512 = hashlib.sha512
    _algorithms = hashlib.algorithms_available

    def file_hash(fname, alg="sha256"):
        """Return a hexadecimal hash of a file read through the binary path API."""
        if type(alg) is str and alg == "sha256":
            hasher = _sha256()
        elif type(alg) is str and alg == "sha512":
            hasher = _sha512()
        else:
            if alg not in _algorithms:
                raise ValueError(f"Algorithm '{alg}' not available in hashlib")
            hasher = hashlib.new(alg)
        with open(fname, "rb", 0) as stream:
            while block := stream.read(262144):
                hasher.update(block)
        return hasher.hexdigest()


__all__ = ["file_hash"]
