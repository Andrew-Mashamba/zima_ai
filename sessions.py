"""
Session Persistence for Zima

Provides SQLite-based storage for:
- Conversation history
- Session metadata
- Resume functionality

Inspired by Claude Code's session management.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

from ollama_client import Message


# Default database location
DEFAULT_DB_DIR = Path.home() / ".config" / "zima"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "sessions.db"


@dataclass
class Session:
    """Represents a chat session."""
    id: str
    created_at: str
    updated_at: str
    working_dir: str
    model: str
    title: Optional[str] = None
    message_count: int = 0
    summary: Optional[str] = None

    @classmethod
    def new(cls, working_dir: str, model: str, title: Optional[str] = None) -> "Session":
        """Create a new session."""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4())[:8],  # Short IDs like Claude Code
            created_at=now,
            updated_at=now,
            working_dir=working_dir,
            model=model,
            title=title,
            message_count=0,
        )


@dataclass
class StoredMessage:
    """A message stored in the database."""
    id: int
    session_id: str
    role: str
    content: str
    created_at: str
    tool_calls: Optional[str] = None  # JSON string of tool calls

    def to_message(self) -> Message:
        """Convert to Message object."""
        return Message(role=self.role, content=self.content)


class SessionStore:
    """
    SQLite-based session storage.

    Usage:
        store = SessionStore()

        # Create new session
        session = store.create_session(working_dir="/path/to/project", model="qwen2.5-coder:3b")

        # Add messages
        store.add_message(session.id, Message(role="user", content="Hello"))
        store.add_message(session.id, Message(role="assistant", content="Hi!"))

        # Resume session
        session = store.get_session(session.id)
        messages = store.get_messages(session.id)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    working_dir TEXT NOT NULL,
                    model TEXT NOT NULL,
                    title TEXT,
                    message_count INTEGER DEFAULT 0,
                    summary TEXT
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    tool_calls TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # Indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_working_dir
                ON sessions(working_dir)
            """)

            conn.commit()

    def create_session(
        self,
        working_dir: str,
        model: str,
        title: Optional[str] = None
    ) -> Session:
        """Create a new session."""
        session = Session.new(working_dir, model, title)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, created_at, updated_at, working_dir, model, title, message_count, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.created_at,
                session.updated_at,
                session.working_dir,
                session.model,
                session.title,
                session.message_count,
                session.summary,
            ))
            conn.commit()

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()

            if row:
                return Session(
                    id=row["id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    working_dir=row["working_dir"],
                    model=row["model"],
                    title=row["title"],
                    message_count=row["message_count"],
                    summary=row["summary"],
                )
        return None

    def update_session(self, session: Session):
        """Update session metadata."""
        session.updated_at = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET updated_at = ?, title = ?, message_count = ?, summary = ?, model = ?
                WHERE id = ?
            """, (
                session.updated_at,
                session.title,
                session.message_count,
                session.summary,
                session.model,
                session.id,
            ))
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_sessions(
        self,
        working_dir: Optional[str] = None,
        limit: int = 20
    ) -> list[Session]:
        """List recent sessions, optionally filtered by working directory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if working_dir:
                cursor.execute("""
                    SELECT * FROM sessions
                    WHERE working_dir = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (working_dir, limit))
            else:
                cursor.execute("""
                    SELECT * FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))

            return [
                Session(
                    id=row["id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    working_dir=row["working_dir"],
                    model=row["model"],
                    title=row["title"],
                    message_count=row["message_count"],
                    summary=row["summary"],
                )
                for row in cursor.fetchall()
            ]

    def get_last_session(self, working_dir: Optional[str] = None) -> Optional[Session]:
        """Get the most recent session."""
        sessions = self.list_sessions(working_dir=working_dir, limit=1)
        return sessions[0] if sessions else None

    def add_message(
        self,
        session_id: str,
        message: Message,
        tool_calls: Optional[list[dict]] = None
    ) -> int:
        """Add a message to a session. Returns message ID."""
        now = datetime.now().isoformat()
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert message
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, created_at, tool_calls)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, message.role, message.content, now, tool_calls_json))

            message_id = cursor.lastrowid

            # Update session message count and timestamp
            cursor.execute("""
                UPDATE sessions
                SET message_count = message_count + 1, updated_at = ?
                WHERE id = ?
            """, (now, session_id))

            # Auto-generate title from first user message
            cursor.execute("""
                UPDATE sessions
                SET title = COALESCE(title, SUBSTR(?, 1, 50))
                WHERE id = ? AND title IS NULL
            """, (message.content if message.role == "user" else "", session_id))

            conn.commit()
            return message_id

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[Message]:
        """Get all messages for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if limit:
                cursor.execute("""
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                """, (session_id,))

            return [
                Message(role=row["role"], content=row["content"])
                for row in cursor.fetchall()
            ]

    def get_stored_messages(self, session_id: str) -> list[StoredMessage]:
        """Get all messages with metadata for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))

            return [
                StoredMessage(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    tool_calls=row["tool_calls"],
                )
                for row in cursor.fetchall()
            ]

    def search_sessions(self, query: str, limit: int = 10) -> list[Session]:
        """Search sessions by title or content."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT s.* FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.title LIKE ? OR m.content LIKE ?
                ORDER BY s.updated_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))

            return [
                Session(
                    id=row["id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    working_dir=row["working_dir"],
                    model=row["model"],
                    title=row["title"],
                    message_count=row["message_count"],
                    summary=row["summary"],
                )
                for row in cursor.fetchall()
            ]

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than specified days. Returns count deleted."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get session IDs to delete
            cursor.execute("""
                SELECT id FROM sessions WHERE updated_at < ?
            """, (cutoff,))
            session_ids = [row["id"] for row in cursor.fetchall()]

            if session_ids:
                # Delete messages first
                cursor.execute(f"""
                    DELETE FROM messages
                    WHERE session_id IN ({','.join('?' * len(session_ids))})
                """, session_ids)

                # Delete sessions
                cursor.execute(f"""
                    DELETE FROM sessions
                    WHERE id IN ({','.join('?' * len(session_ids))})
                """, session_ids)

                conn.commit()

            return len(session_ids)


# Convenience function
def get_session_store() -> SessionStore:
    """Get the default session store."""
    return SessionStore()


if __name__ == "__main__":
    # Test the session store
    print("Testing SessionStore...")

    store = SessionStore()

    # Create a session
    session = store.create_session(
        working_dir="/test/project",
        model="qwen2.5-coder:3b"
    )
    print(f"Created session: {session.id}")

    # Add messages
    store.add_message(session.id, Message(role="user", content="Hello, can you help me?"))
    store.add_message(session.id, Message(role="assistant", content="Of course! What do you need help with?"))
    store.add_message(session.id, Message(role="user", content="Write a hello world function"))

    # Get messages
    messages = store.get_messages(session.id)
    print(f"\nMessages ({len(messages)}):")
    for msg in messages:
        print(f"  [{msg.role}]: {msg.content[:50]}...")

    # List sessions
    sessions = store.list_sessions()
    print(f"\nRecent sessions ({len(sessions)}):")
    for s in sessions:
        print(f"  [{s.id}] {s.title or 'Untitled'} ({s.message_count} messages)")

    # Resume session
    resumed = store.get_session(session.id)
    print(f"\nResumed session: {resumed.id}, messages: {resumed.message_count}")

    print("\nSession persistence working!")
