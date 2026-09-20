"""Source-routing analysis by default; the old autoencoder is explicit."""

import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["head-roles"]:
        from .head_roles.run import main as entry
        entry(args[1:])
    elif args[:1] == ["geometry"]:
        from .head_geometry.run import main as entry
        entry(args[1:])
    elif args[:1] == ["regime"]:
        from .latent_regime.run import main as entry
        entry(args[1:])
    elif args[:1] == ["compatibility"]:
        from .structured_compatibility.run import main as entry
        entry(args[1:])
    elif args[:1] == ["autoencoder"]:
        from .autoencoder_run import main as entry
        entry(args[1:])
    elif args[:1] == ["evaluate"]:
        from .reanchor_evaluate import main as entry
        entry(args[1:])
    else:
        from .reanchor_run import main as entry
        entry(args)


if __name__ == "__main__":
    main()
