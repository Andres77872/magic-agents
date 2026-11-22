"""Codex triggers match whole tokens, including ones with non-word edges."""
import pytest

from magic_agents.node_system.NodeCodex import NodeCodex


@pytest.mark.parametrize('trigger,text,expected', [
    ('python', 'i like python a lot', True),
    ('python', 'pythonic code', False),
    ('python', 'cpython internals', False),
    ('c++', 'use c++ now', True),
    ('c++', 'c++', True),
    ('c++', 'c++x is new', False),
    ('.net', 'built on .net today', True),
    ('node.js', 'deploy with node.js.', True),
    ('node.js', 'node.jsx files', False),
])
def test_whole_token_matching(trigger, text, expected):
    assert NodeCodex._word_boundary_match(trigger, text.casefold()) is expected
