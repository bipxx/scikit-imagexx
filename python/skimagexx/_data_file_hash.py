"""Local file digests through Pooch or the standard library."""

try:
    from pooch import file_hash
except ModuleNotFoundError:
    import hashlib

    def file_hash(fname, alg="sha256"):
        """Return a hexadecimal hash of a file read through the binary path API."""
        if alg not in hashlib.algorithms_available:
            raise ValueError(f"Algorithm '{alg}' not available in hashlib")
        hasher = hashlib.new(alg)
        with open(fname, "rb") as stream:
            while block := stream.read(65536):
                hasher.update(block)
        return hasher.hexdigest()


__all__ = ["file_hash"]
