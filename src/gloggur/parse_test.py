from tree_sitter_language_pack import get_parser

parser = get_parser("python")
tree = parser.parse(b"""
def test_foo():
    assert 1 == 1
    assert "foo" in bar
""")


def walk(n, indent=0):
    print(" " * indent + f"{n.type} ({n.start_byte}-{n.end_byte})")
    for c in n.children:
        walk(c, indent + 2)


walk(tree.root_node)
