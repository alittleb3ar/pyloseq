import pyloseq


def test_importable() -> None:
    assert pyloseq.__version__


def test_version_is_string() -> None:
    assert isinstance(pyloseq.__version__, str)
