"""QdrantVectorDB is an optional extra; without it the failure must say how to install it.

Kept apart from test_qdrant_vector_db.py, which skips entirely when qdrant-client
is absent (as in the default ``[test]`` install), so this runs in every CI job.
"""

from unittest.mock import MagicMock, patch

import pytest

# The package only re-exports QdrantVectorDB when qdrant-client is installed;
# the module itself always defines it so the constructor can explain the extra.
from magic_agents.vector_storage import qdrant_vector_db as qdrant_module
from magic_agents.vector_storage.qdrant_vector_db import QdrantVectorDB


def test_missing_qdrant_client_raises_an_actionable_import_error():
    with patch.object(qdrant_module, "_QDRANT_AVAILABLE", False):
        with pytest.raises(ImportError, match=r"qdrant-client is required") as error:
            QdrantVectorDB(client=MagicMock(), dim=384)

    assert "[qdrant]" in str(error.value)
