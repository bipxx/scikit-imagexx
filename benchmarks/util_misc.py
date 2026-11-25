"""Fixed public constructor and warm documentation-search cases."""


def fixed_public_calls():
    yield 'warning-construction', 'PendingSkimage2Change', ('API will change',), {}
    yield 'docstring-hit', 'lookfor', ('regular_grid',), {}
    yield 'docstring-no-match', 'lookfor', ('no_such_skimage_docstring_term_119',), {}
    yield 'docstring-two-words', 'lookfor', ('regionprops labels',), {}
