# Author: Bradley R. Kinnard
"""
User storage - in-memory for now, can be swapped for SQLite.
"""

import asyncio
from typing import Dict, Optional
from uuid import UUID

from ..core.models.user import User


class UserStore:
    """In-memory user store."""

    def __init__(self):
        self._users: Dict[UUID, User] = {}
        self._email_index: Dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def create(self, user: User) -> User:
        async with self._lock:
            if user.email.lower() in self._email_index:
                raise ValueError(f"Email {user.email} already registered")
            self._users[user.id] = user
            self._email_index[user.email.lower()] = user.id
            return user

    async def get(self, user_id: UUID) -> Optional[User]:
        return self._users.get(user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        user_id = self._email_index.get(email.lower())
        if user_id:
            return self._users.get(user_id)
        return None

    async def update(self, user: User) -> User:
        async with self._lock:
            if user.id not in self._users:
                raise ValueError(f"User {user.id} not found")
            old_user = self._users[user.id]
            # Update email index if email changed
            if old_user.email.lower() != user.email.lower():
                del self._email_index[old_user.email.lower()]
                self._email_index[user.email.lower()] = user.id
            self._users[user.id] = user
            return user

    async def delete(self, user_id: UUID) -> bool:
        async with self._lock:
            user = self._users.get(user_id)
            if user:
                del self._users[user_id]
                del self._email_index[user.email.lower()]
                return True
            return False

    async def list_all(self) -> list[User]:
        return list(self._users.values())

    async def count(self) -> int:
        return len(self._users)


# Singleton
_user_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


__all__ = ["UserStore", "get_user_store"]
