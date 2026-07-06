import asyncio
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


class AsyncCompatibleSqliteSaver(SqliteSaver):
    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config,
        *,
        filter=None,
        before=None,
        limit=None,
    ):
        checkpoints = await asyncio.to_thread(
            lambda: list(
                self.list(
                    config,
                    filter=filter,
                    before=before,
                    limit=limit,
                )
            )
        )
        for checkpoint in checkpoints:
            yield checkpoint

    async def aput(
        self,
        config,
        checkpoint,
        metadata,
        new_versions,
    ):
        return await asyncio.to_thread(
            self.put,
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config,
        writes,
        task_id,
        task_path="",
    ):
        await asyncio.to_thread(
            self.put_writes,
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id):
        await asyncio.to_thread(self.delete_thread, thread_id)

    def close(self):
        self.conn.close()


def create_sqlite_checkpointer(path: str) -> AsyncCompatibleSqliteSaver:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        checkpoint_path,
        check_same_thread=False,
    )
    serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    return AsyncCompatibleSqliteSaver(connection, serde=serializer)