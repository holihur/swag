package llm

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const sampleSwagger = `{
  "swagger": "2.0",
  "info": {
    "title": "Pet Store",
    "description": "A sample API",
    "version": "1.0.0"
  },
  "paths": {
    "/pets": {
      "get": {
        "summary": "List pets",
        "description": "Returns all pets",
        "parameters": [
          {"name": "limit", "in": "query", "description": "Maximum number of pets", "type": "integer"}
        ],
        "responses": {
          "200": {"description": "A list of pets"}
        }
      }
    }
  },
  "definitions": {
    "Pet": {
      "type": "object",
      "properties": {
        "name": {"type": "string", "description": "The pet name"}
      }
    }
  }
}`

func parseSampleSpec(t *testing.T) *spec.Swagger {
	t.Helper()

	var swagger spec.Swagger
	require.NoError(t, json.Unmarshal([]byte(sampleSwagger), &swagger))
	return &swagger
}

type mockClient struct {
	mu       sync.Mutex
	calls    int
	received []Segment
	fn       func(messages []Message) (string, error)
}

func (m *mockClient) Complete(_ context.Context, messages []Message) (string, error) {
	m.mu.Lock()
	m.calls++
	m.mu.Unlock()

	if len(messages) != 2 {
		return "", errors.New("expected two messages")
	}

	if m.fn != nil {
		return m.fn(messages)
	}

	// The user message contains the JSON object of pointers to translate.
	user := messages[1].Content
	start := strings.Index(user, "{")
	if start == -1 {
		return "", errors.New("no payload found")
	}

	var payload map[string]string
	if err := json.Unmarshal([]byte(user[start:]), &payload); err != nil {
		return "", err
	}

	translated := make(map[string]string, len(payload))
	for pointer, text := range payload {
		m.mu.Lock()
		m.received = append(m.received, Segment{Pointer: pointer, Text: text})
		m.mu.Unlock()
		translated[pointer] = "zh:" + text
	}

	encoded, err := json.Marshal(translated)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func (m *mockClient) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func TestExtractSegments(t *testing.T) {
	swagger := parseSampleSpec(t)

	raw, err := json.Marshal(swagger)
	require.NoError(t, err)

	var doc map[string]any
	require.NoError(t, json.Unmarshal(raw, &doc))

	segments := ExtractSegments(doc)

	pointers := make([]string, 0, len(segments))
	for _, segment := range segments {
		pointers = append(pointers, segment.Pointer)
	}

	assert.Contains(t, pointers, "/info/title")
	assert.Contains(t, pointers, "/info/description")
	assert.Contains(t, pointers, "/paths/~1pets/get/summary")
	assert.Contains(t, pointers, "/paths/~1pets/get/description")
	assert.Contains(t, pointers, "/paths/~1pets/get/parameters/0/description")
	assert.Contains(t, pointers, "/paths/~1pets/get/responses/200/description")
	assert.Contains(t, pointers, "/definitions/Pet/properties/name/description")

	// Version must never be translated.
	assert.NotContains(t, pointers, "/info/version")
}

func TestTranslateSpec(t *testing.T) {
	client := &mockClient{}

	localized, err := TranslateSpec(context.Background(), client, parseSampleSpec(t), "zh-CN", TranslateOptions{})
	require.NoError(t, err)

	assert.Equal(t, "zh:Pet Store", localized.Info.Title)
	assert.Equal(t, "zh:A sample API", localized.Info.Description)
	assert.Equal(t, "1.0.0", localized.Info.Version)
	assert.Equal(t, "zh:List pets", localized.Paths.Paths["/pets"].Get.Summary)
	assert.Equal(t, "zh:A list of pets", localized.Paths.Paths["/pets"].Get.Responses.StatusCodeResponses[200].Description)

	// The original document must not be mutated.
	original := parseSampleSpec(t)
	assert.Equal(t, "Pet Store", original.Info.Title)
}

func TestTranslateSpecIncrementalCache(t *testing.T) {
	cache := NewMemoryCache()
	opts := TranslateOptions{Cache: cache}
	client := &mockClient{}

	// First run translates everything.
	_, err := TranslateSpec(context.Background(), client, parseSampleSpec(t), "zh-CN", opts)
	require.NoError(t, err)
	assert.Positive(t, client.callCount())

	firstReceived := len(client.received)
	assert.Positive(t, firstReceived)

	// Reset the recorder and run again with the same document. Everything
	// should be served from the cache without contacting the LLM.
	client.received = nil
	client.calls = 0

	_, err = TranslateSpec(context.Background(), client, parseSampleSpec(t), "zh-CN", opts)
	require.NoError(t, err)
	assert.Equal(t, 0, client.callCount(), "unchanged document must not call the LLM")

	// Only the changed string should be sent on the next run.
	modified := parseSampleSpec(t)
	modified.Info.Title = "Pet Store API"

	client.received = nil
	client.calls = 0

	_, err = TranslateSpec(context.Background(), client, modified, "zh-CN", opts)
	require.NoError(t, err)
	assert.Equal(t, 1, client.callCount())

	client.mu.Lock()
	received := append([]Segment(nil), client.received...)
	client.mu.Unlock()

	require.Len(t, received, 1)
	assert.Equal(t, "Pet Store API", received[0].Text)

	// The number of cached entries must not have shrunk.
	assert.GreaterOrEqual(t, cache.Len(), firstReceived)
}

func TestTranslateSegmentsDifferentLanguages(t *testing.T) {
	cache := NewMemoryCache()
	client := &mockClient{}

	segments := []Segment{{Pointer: "/info/title", Text: "Pet Store"}}

	_, err := TranslateSegments(context.Background(), client, segments, "zh-CN", TranslateOptions{Cache: cache})
	require.NoError(t, err)

	// The same source string in another language must not hit the cache.
	_, err = TranslateSegments(context.Background(), client, segments, "ja", TranslateOptions{Cache: cache})
	require.NoError(t, err)

	assert.Equal(t, 2, cache.Len())
}

func TestCacheKeyFingerprint(t *testing.T) {
	base := TranslateOptions{}
	other := TranslateOptions{Glossary: map[string]string{"Pet": "宠物"}}

	assert.NotEqual(t, CacheKey("zh-CN", base.fingerprint(), "Pet Store"), CacheKey("ja", base.fingerprint(), "Pet Store"))
	assert.NotEqual(t, CacheKey("zh-CN", base.fingerprint(), "Pet Store"), CacheKey("zh-CN", other.fingerprint(), "Pet Store"))
	assert.Equal(t, CacheKey("zh-CN", base.fingerprint(), "Pet Store"), CacheKey("zh-CN", base.fingerprint(), "Pet Store"))
}

func TestBatchSegments(t *testing.T) {
	segments := []Segment{
		{Text: "12345"},
		{Text: "12345"},
		{Text: "12345"},
	}

	batches := batchSegments(segments, 2, 1000)
	require.Len(t, batches, 2)
	assert.Len(t, batches[0], 2)
	assert.Len(t, batches[1], 1)

	batches = batchSegments(segments, 100, 6)
	require.Len(t, batches, 3)
	for _, batch := range batches {
		assert.Len(t, batch, 1)
	}
}

func TestParseTranslationResponse(t *testing.T) {
	tt := []struct {
		name     string
		raw      string
		expected map[string]string
		wantErr  bool
	}{
		{
			name:     "plain json",
			raw:      `{"/info/title":"宠物商店"}`,
			expected: map[string]string{"/info/title": "宠物商店"},
		},
		{
			name:     "fenced json",
			raw:      "```json\n{\"/info/title\":\"宠物商店\"}\n```",
			expected: map[string]string{"/info/title": "宠物商店"},
		},
		{
			name:    "garbage",
			raw:     "no json here",
			wantErr: true,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseTranslationResponse(tc.raw)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.expected, got)
		})
	}
}

func TestSetByPointer(t *testing.T) {
	doc := map[string]any{
		"info": map[string]any{"title": "old"},
		"list": []any{map[string]any{"description": "old"}},
	}

	require.NoError(t, setByPointer(doc, "/info/title", "new"))
	require.NoError(t, setByPointer(doc, "/list/0/description", "new"))

	assert.Equal(t, "new", doc["info"].(map[string]any)["title"])
	assert.Equal(t, "new", doc["list"].([]any)[0].(map[string]any)["description"])

	require.Error(t, setByPointer(doc, "/info/missing", "x"))
	require.Error(t, setByPointer(doc, "/list/5/description", "x"))
}

func TestFileCache(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")

	cache, err := NewFileCache(path)
	require.NoError(t, err)
	assert.False(t, cache.Dirty())

	cache.Set("key", "value")
	assert.Equal(t, 1, cache.Len())
	assert.True(t, cache.Dirty())

	require.NoError(t, cache.Save())
	assert.False(t, cache.Dirty())

	_, err = os.Stat(path)
	require.NoError(t, err)

	reloaded, err := NewFileCache(path)
	require.NoError(t, err)

	value, ok := reloaded.Get("key")
	assert.True(t, ok)
	assert.Equal(t, "value", value)

	// Saving without changes is a no-op.
	require.NoError(t, reloaded.Save())
}

func TestFileCacheMissing(t *testing.T) {
	cache, err := NewFileCache(filepath.Join(t.TempDir(), "does-not-exist.json"))
	require.NoError(t, err)
	assert.Equal(t, 0, cache.Len())
}
