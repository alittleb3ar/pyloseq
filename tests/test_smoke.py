import phyla


def test_importable() -> None:
    assert phyla.__version__


def test_version_is_string() -> None:
    assert isinstance(phyla.__version__, str)
