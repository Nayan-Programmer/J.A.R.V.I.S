"""
TASK MANAGER
============

In-memory store for background tasks (image generation, content writing, etc.).

The frontend polls GET /tasks/{task_id} to check if a task is done.
When complete, the result is stored here so the viewer (viewer.html) can fetch it.

Lifecycle:
  1. task_manager.create_task(type, label) → task_id
  2. asyncio task calls task_manager.complete_task(task_id, result)
     or task_manager.fail_task(task_id, error)
  3. Frontend polls /tasks/{task_id} → {"status": "completed", "result": ...}
  4. Frontend opens viewer.html?task_id=... to display result
"""

import uuid
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("J.A.R.V.I.S")


class TaskManager:
    """
    Thread/async-safe in-memory task store.

    Tasks are kept for the lifetime of the server process.
    For a production system you'd persist these; for single-user local use,
    in-memory is perfectly fine.
    """

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------

    def create_task(self, task_type: str, label: str) -> str:
        """
        Register a new background task and return its unique ID.

        Args:
            task_type: e.g. "generate_image", "content"
            label:     Short description shown in the UI card (e.g. the prompt)

        Returns:
            str: UUID task_id — pass this back to the frontend.
        """
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "task_id":  task_id,
            "status":   "pending",   # pending → completed | failed
            "type":     task_type,
            "label":    label,
            "result":   None,        # set by complete_task()
            "error":    None,        # set by fail_task()
        }
        logger.info("[TASK] Created task %s (%s): %s", task_id[:8], task_type, label[:60])
        return task_id

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return the task dict, or None if the task_id is unknown."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Return all tasks (useful for debugging)."""
        return dict(self._tasks)

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def complete_task(self, task_id: str, result: Any) -> None:
        """
        Mark task as completed and store its result.

        Args:
            task_id: The task's UUID.
            result:  Any JSON-serialisable value (e.g. image URL, HTML string).
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("[TASK] complete_task called for unknown task_id: %s", task_id)
            return
        task["status"] = "completed"
        task["result"] = result
        logger.info("[TASK] Completed task %s", task_id[:8])

    def fail_task(self, task_id: str, error: str) -> None:
        """
        Mark task as failed and record the error message.

        Args:
            task_id: The task's UUID.
            error:   Human-readable error description.
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("[TASK] fail_task called for unknown task_id: %s", task_id)
            return
        task["status"] = "failed"
        task["error"]  = error
        logger.warning("[TASK] Failed task %s: %s", task_id[:8], error)

    def update_status(self, task_id: str, status: str) -> None:
        """Update the status field without touching result/error."""
        task = self._tasks.get(task_id)
        if task:
            task["status"] = status

    # ------------------------------------------------------------------
    # SERIALISE FOR HTTP RESPONSE
    # ------------------------------------------------------------------

    def to_response(self, task_id: str) -> Dict[str, Any]:
        """
        Return a dict suitable for the GET /tasks/{task_id} JSON response.
        Returns a 'not_found' dict if the task_id is unknown.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return {
                "task_id": task_id,
                "status":  "not_found",
                "error":   "Task not found. It may have expired.",
            }
        return {
            "task_id": task["task_id"],
            "status":  task["status"],
            "type":    task["type"],
            "label":   task["label"],
            "result":  task["result"],
            "error":   task["error"],
        }


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this in main.py and route handlers.
task_manager = TaskManager()
