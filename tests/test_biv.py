from biv import hello


def test_hello() -> None:
    assert hello() == "Hello from biv!"
