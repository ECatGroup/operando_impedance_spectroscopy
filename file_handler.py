import queue

from watchdog.events import FileSystemEventHandler

import logging
import time
import os


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, q: queue.Queue):
        self.file_queue = q

    def on_created(self, event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a new file is created.
            self.wait_until_stable(event.src_path)
            self.file_queue.put(event.src_path)
            logging.info(f'New file detected: {event.src_path}')

    def wait_until_stable(self, file_path, timeout=1, interval=0.2):
        """Waits until the file size hasn't changed for `timeout` seconds."""
        last_size = -1
        stable_for = 0

        while stable_for < timeout:
            try:
                current_size = os.path.getsize(file_path)
            except OSError:
                current_size = -1

            if current_size != last_size:
                last_size = current_size
                stable_for = 0
            else:
                stable_for += interval

            time.sleep(interval)
