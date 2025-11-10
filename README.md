# "Uncategorized" miscellaneous utilities.

This is not meant to be an independent package and will not be published on PyPI.
Instead, it is expected to be imported as an "editable" dependency during development.
Once development stabilizes, the code is expected to be copied directly in the client code.

Not intended for use by anyone other than myself :)

## health keeping

### test

```
uv run pytest
```

### code format

```
uv tool run ruff check
```

### type check

```
uv tool run mypy miscutils
```

