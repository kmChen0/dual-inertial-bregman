"""Prevent simultaneous benchmark timing within this repository."""
import os
try:
    import fcntl
except ImportError:
    import msvcrt
    class fcntl:
        LOCK_EX = 1
        LOCK_UN = 2
        @staticmethod
        def flock(stream, operation):
            stream.seek(0)
            if operation == fcntl.LOCK_EX:
                if os.fstat(stream.fileno()).st_size == 0:
                    stream.write('0')
                    stream.flush()
                stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK if operation == fcntl.LOCK_UN else msvcrt.LK_LOCK, 1)
