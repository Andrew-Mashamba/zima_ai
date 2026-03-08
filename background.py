"""
Background Agents for Zima

Allows tasks to run asynchronously without blocking the main conversation.

Features:
- Start long-running tasks in the background
- Check task status
- Retrieve results when complete
- Cancel running tasks

Inspired by Cursor's background agents.
"""

import threading
import queue
import uuid
import time
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """A background task."""
    id: str
    name: str
    description: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    thread: Optional[threading.Thread] = None


@dataclass
class TaskResult:
    """Result from a background task."""
    task_id: str
    success: bool
    output: str
    error: Optional[str] = None
    duration: float = 0.0


class BackgroundTaskManager:
    """
    Manages background task execution.

    Usage:
        manager = BackgroundTaskManager()

        # Start a task
        task_id = manager.start_task(
            name="Build project",
            description="Running npm build",
            func=lambda: run_command("npm build"),
        )

        # Check status
        task = manager.get_task(task_id)
        print(f"Status: {task.status}")

        # Get result when done
        if task.status == TaskStatus.COMPLETED:
            print(f"Result: {task.result}")
    """

    def __init__(self, max_concurrent: int = 3):
        self.tasks: dict[str, BackgroundTask] = {}
        self.max_concurrent = max_concurrent
        self._lock = threading.Lock()

    def _generate_id(self) -> str:
        """Generate a short unique ID."""
        return str(uuid.uuid4())[:8]

    def _get_running_count(self) -> int:
        """Get number of currently running tasks."""
        with self._lock:
            return sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)

    def start_task(
        self,
        name: str,
        description: str,
        func: Callable[[], str],
        on_complete: Optional[Callable[[TaskResult], None]] = None
    ) -> str:
        """
        Start a background task.

        Args:
            name: Short task name
            description: Task description
            func: Function to execute (should return a string result)
            on_complete: Optional callback when task completes

        Returns:
            Task ID
        """
        if self._get_running_count() >= self.max_concurrent:
            raise RuntimeError(f"Max concurrent tasks ({self.max_concurrent}) reached")

        task_id = self._generate_id()
        now = datetime.now().isoformat()

        task = BackgroundTask(
            id=task_id,
            name=name,
            description=description,
            status=TaskStatus.PENDING,
            created_at=now,
        )

        with self._lock:
            self.tasks[task_id] = task

        # Create and start thread
        def run_task():
            start_time = time.time()

            with self._lock:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now().isoformat()

            try:
                result = func()

                with self._lock:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now().isoformat()
                    task.result = result
                    task.progress = 100

                duration = time.time() - start_time

                if on_complete:
                    on_complete(TaskResult(
                        task_id=task_id,
                        success=True,
                        output=result,
                        duration=duration
                    ))

            except Exception as e:
                with self._lock:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now().isoformat()
                    task.error = str(e)

                duration = time.time() - start_time

                if on_complete:
                    on_complete(TaskResult(
                        task_id=task_id,
                        success=False,
                        output="",
                        error=str(e),
                        duration=duration
                    ))

        thread = threading.Thread(target=run_task, daemon=True)
        task.thread = thread
        thread.start()

        return task_id

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        with self._lock:
            return self.tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> list[BackgroundTask]:
        """List all tasks, optionally filtered by status."""
        with self._lock:
            tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Note: This only marks the task as cancelled. The thread may continue
        running if it doesn't check for cancellation.
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now().isoformat()
                return True

        return False

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Wait for a task to complete."""
        task = self.get_task(task_id)
        if not task or not task.thread:
            return None

        task.thread.join(timeout=timeout)

        task = self.get_task(task_id)
        if task.status == TaskStatus.COMPLETED:
            return TaskResult(
                task_id=task_id,
                success=True,
                output=task.result or ""
            )
        elif task.status == TaskStatus.FAILED:
            return TaskResult(
                task_id=task_id,
                success=False,
                output="",
                error=task.error
            )

        return None

    def cleanup_old(self, max_age_hours: int = 24):
        """Remove old completed tasks."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        with self._lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    created = datetime.fromisoformat(task.created_at).timestamp()
                    if created < cutoff:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]


class BackgroundAgentRunner:
    """
    Runs agent tasks in the background.

    Usage:
        runner = BackgroundAgentRunner(agent)

        # Start a background task
        task_id = runner.run_in_background(
            "Explore the codebase and find all API endpoints"
        )

        # Check status
        status = runner.get_status(task_id)

        # Get result
        result = runner.get_result(task_id)
    """

    def __init__(self, agent_factory: Callable, working_dir: Optional[str] = None):
        """
        Args:
            agent_factory: Function that creates a new agent instance
            working_dir: Working directory for agents
        """
        self.agent_factory = agent_factory
        self.working_dir = working_dir
        self.task_manager = BackgroundTaskManager()

    def run_in_background(
        self,
        prompt: str,
        name: Optional[str] = None,
        on_complete: Optional[Callable[[TaskResult], None]] = None
    ) -> str:
        """
        Run a prompt in the background.

        Args:
            prompt: The prompt to execute
            name: Optional task name

        Returns:
            Task ID
        """
        task_name = name or prompt[:30] + ("..." if len(prompt) > 30 else "")

        def execute():
            agent = self.agent_factory()
            return agent.chat(prompt)

        return self.task_manager.start_task(
            name=task_name,
            description=prompt,
            func=execute,
            on_complete=on_complete
        )

    def get_status(self, task_id: str) -> Optional[dict]:
        """Get task status."""
        task = self.task_manager.get_task(task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "progress": task.progress,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }

    def get_result(self, task_id: str) -> Optional[str]:
        """Get task result if completed."""
        task = self.task_manager.get_task(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None

    def list_tasks(self) -> list[dict]:
        """List all background tasks."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "status": t.status.value,
                "progress": t.progress,
            }
            for t in self.task_manager.list_tasks()
        ]

    def cancel(self, task_id: str) -> bool:
        """Cancel a background task."""
        return self.task_manager.cancel_task(task_id)

    def wait(self, task_id: str, timeout: float = 60) -> Optional[TaskResult]:
        """Wait for a task to complete."""
        return self.task_manager.wait_for_task(task_id, timeout)


if __name__ == "__main__":
    import time

    print("Testing BackgroundTaskManager...")

    manager = BackgroundTaskManager()

    # Start a test task
    def slow_task():
        time.sleep(1)
        return "Task completed!"

    task_id = manager.start_task(
        name="Test task",
        description="A slow test task",
        func=slow_task
    )

    print(f"Started task: {task_id}")

    # Check status
    task = manager.get_task(task_id)
    print(f"Initial status: {task.status.value}")

    # Wait for completion
    result = manager.wait_for_task(task_id, timeout=5)

    if result:
        print(f"Result: {result.output}")
        print(f"Success: {result.success}")

    # List tasks
    print("\nAll tasks:")
    for t in manager.list_tasks():
        print(f"  [{t.id}] {t.name}: {t.status.value}")

    print("\n✓ Background agents working!")
