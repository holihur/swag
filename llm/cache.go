package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Cache stores previously computed translations so that only new or changed
// strings need to be sent to the LLM. This makes repeated generation runs
// incremental.
type Cache interface {
	// Get returns the cached translation for the given key.
	Get(key string) (string, bool)
	// Set stores a translation for the given key.
	Set(key, value string)
}

// MemoryCache is a simple in-memory Cache implementation.
type MemoryCache struct {
	mu      sync.RWMutex
	entries map[string]string
}

// NewMemoryCache creates an empty in-memory cache.
func NewMemoryCache() *MemoryCache {
	return &MemoryCache{entries: make(map[string]string)}
}

// Get implements Cache.
func (c *MemoryCache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, ok := c.entries[key]
	return value, ok
}

// Set implements Cache.
func (c *MemoryCache) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[key] = value
}

// Len returns the number of cached entries.
func (c *MemoryCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

const cacheFileVersion = 1

type fileCacheData struct {
	Version int               `json:"version"`
	Entries map[string]string `json:"entries"`
}

// FileCache is a Cache backed by a JSON file. It allows translations to be
// reused across separate swag invocations.
type FileCache struct {
	mu      sync.RWMutex
	path    string
	entries map[string]string
	dirty   bool
}

// NewFileCache loads a cache from the given file. A missing file results in an
// empty cache. The file is created when Save is called.
func NewFileCache(path string) (*FileCache, error) {
	cache := &FileCache{
		path:    path,
		entries: make(map[string]string),
	}
	if path == "" {
		return cache, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cache, nil
		}
		return nil, fmt.Errorf("llm: read cache %q: %w", path, err)
	}

	if len(data) == 0 {
		return cache, nil
	}

	var decoded fileCacheData
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("llm: parse cache %q: %w", path, err)
	}
	for key, value := range decoded.Entries {
		cache.entries[key] = value
	}

	return cache, nil
}

// Get implements Cache.
func (c *FileCache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, ok := c.entries[key]
	return value, ok
}

// Set implements Cache.
func (c *FileCache) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if existing, ok := c.entries[key]; ok && existing == value {
		return
	}
	c.entries[key] = value
	c.dirty = true
}

// Len returns the number of cached entries.
func (c *FileCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// Dirty reports whether the cache has unsaved changes.
func (c *FileCache) Dirty() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.dirty
}

// Save persists the cache to disk. It is a no-op when the cache is not dirty
// or when no path was configured.
func (c *FileCache) Save() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.path == "" || !c.dirty {
		return nil
	}

	data, err := json.MarshalIndent(fileCacheData{
		Version: cacheFileVersion,
		Entries: c.entries,
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("llm: marshal cache: %w", err)
	}

	dir := filepath.Dir(c.path)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return fmt.Errorf("llm: create cache dir: %w", err)
	}

	tmp, err := os.CreateTemp(dir, ".swag-llm-cache-*")
	if err != nil {
		return fmt.Errorf("llm: create temp cache file: %w", err)
	}
	tmpName := tmp.Name()

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("llm: write temp cache file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("llm: close temp cache file: %w", err)
	}
	if err := os.Rename(tmpName, c.path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("llm: replace cache file: %w", err)
	}

	c.dirty = false
	return nil
}
