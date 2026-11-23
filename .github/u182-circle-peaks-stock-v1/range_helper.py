"""Exact bounded range transport classes from the admitted Hough carrier."""
import io
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteZip(io.RawIOBase):
    def __init__(self, url, length):
        super().__init__()
        self.url, self.length, self.position = url, length, 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.length}[whence]
        self.position = base + offset
        if self.position < 0:
            raise OSError('negative remote ZIP offset')
        return self.position

    def read(self, size=-1):
        remaining = max(self.length - self.position, 0)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b''
        if size > 32 * 1024**2 or self.bytes_read + size > 160 * 1024**2:
            raise RuntimeError('diagnostic slice exceeds the retained-data limit')
        start, end = self.position, self.position + size - 1
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                assert response.status == 206, 'blob did not honor a bounded range'
                assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{self.length}'
                raw = response.read(size + 1)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Artifact blob range HTTP {error.code}') from None
        assert len(raw) == size
        self.position += size
        self.bytes_read += size
        return raw
