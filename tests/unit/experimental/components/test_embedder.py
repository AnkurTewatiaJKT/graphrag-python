#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.types import (
    TextChunk,
    TextChunks,
)


@pytest.mark.asyncio
async def test_text_chunk_embedder_run(embedder: MagicMock) -> None:
    embedder.async_embed_query.return_value = [1.0, 2.0, 3.0]
    text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
    text_chunks = TextChunks(
        chunks=[TextChunk(text="may thy knife chip and shatter", index=0)]
    )
    embedded_chunks = await text_chunk_embedder.run(text_chunks)
    embedder.async_embed_query.assert_called_once_with("may thy knife chip and shatter")
    assert isinstance(embedded_chunks, TextChunks)
    for chunk in embedded_chunks.chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.metadata is not None
        assert "embedding" in chunk.metadata.keys()
        assert isinstance(chunk.metadata["embedding"], list)
        for i in chunk.metadata["embedding"]:
            assert isinstance(i, float)


@pytest.mark.asyncio
async def test_text_chunk_embedder_run_skips_failed_chunks(embedder: MagicMock) -> None:
    embedder.async_embed_query.side_effect = [
        [1.0, 2.0, 3.0],
        EmbeddingsGenerationError("rate limit"),
    ]
    text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
    text_chunks = TextChunks(
        chunks=[
            TextChunk(text="may thy knife chip and shatter", index=0),
            TextChunk(text="fear is the mind-killer", index=1),
        ]
    )

    embedded_chunks = await text_chunk_embedder.run(text_chunks)

    assert len(embedded_chunks.chunks) == 1
    assert embedded_chunks.chunks[0].index == 0
    assert embedded_chunks.chunks[0].metadata is not None
    assert "embedding" in embedded_chunks.chunks[0].metadata
