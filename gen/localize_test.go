package gen

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/swaggo/swag/llm"
)

// prefixClient translates every requested string by prefixing it. It records
// how many strings it was asked to translate.
type prefixClient struct {
	mu    sync.Mutex
	calls int
	count int
}

func (c *prefixClient) Complete(_ context.Context, messages []llm.Message) (string, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()

	user := messages[len(messages)-1].Content
	start := strings.Index(user, "{")
	if start == -1 {
		return "{}", nil
	}

	var payload map[string]string
	if err := json.Unmarshal([]byte(user[start:]), &payload); err != nil {
		return "", err
	}

	translated := make(map[string]string, len(payload))
	for pointer, text := range payload {
		translated[pointer] = "[zh] " + text
	}

	c.mu.Lock()
	c.count += len(payload)
	c.mu.Unlock()

	encoded, err := json.Marshal(translated)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func (c *prefixClient) stats() (calls, count int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls, c.count
}

func TestGen_BuildLocalized(t *testing.T) {
	outputDir := t.TempDir()
	cacheFile := filepath.Join(outputDir, "translation-cache.json")
	client := &prefixClient{}

	config := &Config{
		SearchDir:   "../testdata/simple",
		MainAPIFile: "./main.go",
		OutputDir:   outputDir,
		OutputTypes: []string{"go", "json", "yaml"},
		PackageName: "docs",
		Localization: &LocalizationConfig{
			Languages: []string{"zh-CN"},
			Client:    client,
			CacheFile: cacheFile,
		},
	}

	require.NoError(t, New().Build(config))

	expectedFiles := []string{
		filepath.Join(outputDir, "docs.go"),
		filepath.Join(outputDir, "swagger.json"),
		filepath.Join(outputDir, "swagger.yaml"),
		filepath.Join(outputDir, "docs.zh-CN.go"),
		filepath.Join(outputDir, "swagger.zh-CN.json"),
		filepath.Join(outputDir, "swagger.zh-CN.yaml"),
		cacheFile,
	}
	for _, name := range expectedFiles {
		_, err := os.Stat(name)
		assert.NoErrorf(t, err, "expected %s to exist", name)
	}

	// The localized JSON must contain translated strings.
	localizedJSON, err := os.ReadFile(filepath.Join(outputDir, "swagger.zh-CN.json"))
	require.NoError(t, err)
	assert.Contains(t, string(localizedJSON), "[zh] ")

	// The default JSON must stay untouched.
	defaultJSON, err := os.ReadFile(filepath.Join(outputDir, "swagger.json"))
	require.NoError(t, err)
	assert.NotContains(t, string(defaultJSON), "[zh] ")

	// The localized Go file must register a language specific instance.
	localizedGo, err := os.ReadFile(filepath.Join(outputDir, "docs.zh-CN.go"))
	require.NoError(t, err)
	assert.Contains(t, string(localizedGo), "package docs")
	assert.Contains(t, string(localizedGo), "[zh] ")

	// Incremental run: everything is served from the cache.
	_, countBefore := client.stats()
	require.Positive(t, countBefore)

	client.mu.Lock()
	client.calls = 0
	client.count = 0
	client.mu.Unlock()

	require.NoError(t, New().Build(config))

	calls, count := client.stats()
	assert.Equal(t, 0, calls, "unchanged document must not call the LLM")
	assert.Equal(t, 0, count)
}

func TestGen_BuildLocalizedRequiresClient(t *testing.T) {
	config := &Config{
		SearchDir:   "../testdata/simple",
		MainAPIFile: "./main.go",
		OutputDir:   t.TempDir(),
		OutputTypes: []string{"json"},
		Localization: &LocalizationConfig{
			Languages: []string{"zh-CN"},
		},
	}

	err := New().Build(config)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "client is required")
}

func TestNormalizeLanguage(t *testing.T) {
	assert.Equal(t, "zh_CN", normalizeLanguage("zh-CN"))
	assert.Equal(t, "pt_BR", normalizeLanguage("pt-BR"))
	assert.Equal(t, "ja", normalizeLanguage("ja"))
}

func TestOutputPath(t *testing.T) {
	assert.Equal(t, "out/swagger.json", outputPath(&Config{OutputDir: "out"}, "swagger.json"))
	assert.Equal(t, "out/swagger.zh-CN.json", outputPath(&Config{OutputDir: "out", Language: "zh-CN"}, "swagger.json"))
	assert.Equal(t, "out/docs.zh-CN.go", outputPath(&Config{OutputDir: "out", Language: "zh-CN"}, "docs.go"))
}
