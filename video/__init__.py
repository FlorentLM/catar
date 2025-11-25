"""
Unified video subsystem for CATAR.

Provides video frame access through pluggable backends with support
for direct file access or compressed disk caching.

Public API:
    create_video_backend(): Factory for creating video backends
    DiskCacheBuilder: Tool for building video cache (preprocessing)
    VideoReaderWorker: Worker thread for reading and distributing frames
    BatchVideoReader: Helper for batch frame access

Internal (not exported):
    DiskCacheReader: Low-level cache reader (used internally by CachedBackend)
"""

from video.backends import (
    VideoBackend,
    create_video_backend
)
from video.disk_cache import DiskCacheBuilder
from video.readers import VideoReaderWorker, BatchVideoReader

__all__ = [
    'VideoBackend',
    'create_video_backend',
    'DiskCacheBuilder',      # Public (preprocessing)
    'VideoReaderWorker',
    'BatchVideoReader',
]
# DiskCacheReader intentionally not exposed