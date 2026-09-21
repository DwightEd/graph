"""Import the heavyweight adapter only for model execution."""


def load_model(name, revision="main", device="cpu", dtype="float32"):
    from .adapter import load_model as load

    return load(name, revision, device, dtype)
