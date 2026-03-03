from __future__ import annotations

from gloggur.cli.main import main as cli_main


def main() -> int:
    """Run the gloggur CLI entrypoint."""
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
