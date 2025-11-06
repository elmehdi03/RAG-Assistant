"""
File watcher for monitoring PDF files in the data/ folder.
Automatically re-indexes when new PDFs are detected.
Uses SHA256 checksums to avoid re-processing unchanged files.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion import load_documents_from_pdf
from embeddings import build_faiss_index


# Track processed files with their checksums
CACHE_FILE = "data/.watcher_cache.json"


class FileCache:
    """Manages file processing cache with SHA256 checksums."""
    
    def __init__(self, cache_path=CACHE_FILE):
        self.cache_path = cache_path
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print("⚠️ Cache corrupted, creating new one.")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"⚠️ Error saving cache: {str(e)}")
    
    def _get_file_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            print(f"⚠️ Cannot read {file_path}: {str(e)}")
            return None
    
    def is_file_changed(self, file_path):
        """Check if a file has changed since last processing."""
        # Wait a moment for file to finish writing
        time.sleep(0.5)
        
        file_hash = self._get_file_hash(file_path)
        if file_hash is None:
            return False
        
        cached_hash = self.cache.get(file_path)
        
        if cached_hash != file_hash:
            self.cache[file_path] = file_hash
            self._save_cache()
            return True
        return False
    
    def mark_all_files_processed(self, pdf_files):
        """Mark all PDF files as processed."""
        for pdf_file in pdf_files:
            file_path = os.path.join("data", pdf_file)
            file_hash = self._get_file_hash(file_path)
            if file_hash:
                self.cache[file_path] = file_hash
        self._save_cache()
    
    def get_cached_files_count(self):
        """Get number of cached files."""
        return len(self.cache)
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        self._save_cache()
        print("✅ Cache cleared.")


class PDFWatcherHandler(FileSystemEventHandler):
    """Handles file system events for PDF files."""
    
    def __init__(self, file_cache):
        self.file_cache = file_cache
        self.reindex_in_progress = False

    def on_created(self, event):
        """Triggered when a new file is created."""
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        
        print(f"✅ New PDF detected: {event.src_path}")
        self.trigger_reindex()

    def on_modified(self, event):
        """Triggered when a file is modified."""
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        
        # Skip if file is still being written (permission denied)
        if self.reindex_in_progress:
            print(f"⏭️ Reindex already in progress, skipping: {event.src_path}")
            return
        
        if self.file_cache.is_file_changed(event.src_path):
            print(f"📝 PDF changed: {event.src_path}")
            self.trigger_reindex()
        else:
            print(f"⏭️ PDF unchanged (skipped): {event.src_path}")

    def trigger_reindex(self):
        """Trigger FAISS index rebuild."""
        if self.reindex_in_progress:
            print("⏳ Reindex already in progress...")
            return
        
        self.reindex_in_progress = True
        try:
            print("🔄 Starting re-indexing...")
            texts, metadata = load_documents_from_pdf("data")
            
            if texts:
                build_faiss_index(texts, metadata)
                
                # Update cache for all PDFs
                pdf_files = get_pdf_files()
                self.file_cache.mark_all_files_processed(pdf_files)
                
                print("✅ Re-indexing completed successfully!")
                print(f"📊 Indexed files: {len(texts)} chunks from {len(pdf_files)} PDFs")
                print(f"💾 Cached files: {self.file_cache.get_cached_files_count()}")
            else:
                print("⚠️ No PDF files found in data/ folder.")
        except Exception as e:
            print(f"❌ Error during re-indexing: {str(e)}")
        finally:
            self.reindex_in_progress = False


def get_pdf_files():
    """Get list of all PDF files in data/ folder."""
    data_folder = Path("data")
    if not data_folder.exists():
        return []
    return sorted([f.name for f in data_folder.glob("*.pdf")])


def start_watcher():
    """Start the file watcher."""
    print("=" * 60)
    print("👀 PDF File Watcher Started")
    print("=" * 60)
    print(f"📁 Monitoring: data/ folder")
    
    file_cache = FileCache()
    print(f"💾 Cache loaded: {file_cache.get_cached_files_count()} files")
    
    observer = Observer()
    event_handler = PDFWatcherHandler(file_cache)
    
    # Watch the data/ folder
    observer.schedule(event_handler, path="data", recursive=False)
    observer.start()
    
    print("✅ Watcher is running. Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        while True:
            observer.join(timeout=1)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 Stopping file watcher...")
        observer.stop()
    
    observer.join()
    print("✅ File watcher stopped.")
    print("=" * 60)


if __name__ == "__main__":
    start_watcher()
