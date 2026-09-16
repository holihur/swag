package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClientBuildsURL(t *testing.T) {
	tt := []struct {
		in     string
		expect string
	}{
		{"", DefaultBaseURL + "/chat/completions"},
		{"https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"},
		{"https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"},
		{"https://example.com/chat/completions", "https://example.com/chat/completions"},
	}

	for _, tc := range tt {
		t.Run(tc.in, func(t *testing.T) {
			client := NewClient(ClientConfig{BaseURL: tc.in})
			assert.Equal(t, tc.expect, client.baseURL)
		})
	}
}

func TestClientComplete(t *testing.T) {
	var gotAuth string
	var gotModel string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var payload map[string]any
		require.NoError(t, json.Unmarshal(body, &payload))
		gotModel, _ = payload["model"].(string)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hello"}}]}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		BaseURL: server.URL,
		APIKey:  "secret",
		Model:   "test-model",
	})

	reply, err := client.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}})
	require.NoError(t, err)
	assert.Equal(t, "hello", reply)
	assert.Equal(t, "Bearer secret", gotAuth)
	assert.Equal(t, "test-model", gotModel)
}

func TestClientCompleteErrors(t *testing.T) {
	t.Run("status code", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}))
		defer server.Close()

		client := NewClient(ClientConfig{BaseURL: server.URL, Model: "m"})
		_, err := client.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unexpected status 500")
	})

	t.Run("api error payload", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"error":{"message":"invalid key","type":"auth"}}`))
		}))
		defer server.Close()

		client := NewClient(ClientConfig{BaseURL: server.URL, Model: "m"})
		_, err := client.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid key")
	})

	t.Run("no choices", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"choices":[]}`))
		}))
		defer server.Close()

		client := NewClient(ClientConfig{BaseURL: server.URL, Model: "m"})
		_, err := client.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no choices")
	})
}

func TestNewClientDetectsAnthropic(t *testing.T) {
	client := NewClient(ClientConfig{BaseURL: "https://api.longcat.chat/anthropic"})
	assert.Equal(t, ProtocolAnthropic, client.protocol)
	assert.Equal(t, "https://api.longcat.chat/anthropic/v1/messages", client.baseURL)

	client = NewClient(ClientConfig{BaseURL: "https://api.longcat.chat/anthropic", Protocol: ProtocolOpenAI})
	assert.Equal(t, ProtocolOpenAI, client.protocol)
	assert.Equal(t, "https://api.longcat.chat/anthropic/chat/completions", client.baseURL)
}

func TestClientCompleteAnthropic(t *testing.T) {
	var gotAPIKey string
	var gotAuth string
	var gotVersion string
	var gotSystem string
	var gotMaxTokens float64

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/anthropic/v1/messages", r.URL.Path)
		gotAPIKey = r.Header.Get("x-api-key")
		gotAuth = r.Header.Get("Authorization")
		gotVersion = r.Header.Get("anthropic-version")

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var payload map[string]any
		require.NoError(t, json.Unmarshal(body, &payload))
		gotSystem, _ = payload["system"].(string)
		gotMaxTokens, _ = payload["max_tokens"].(float64)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"pong"}]}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{
		BaseURL: server.URL + "/anthropic",
		APIKey:  "secret",
		Model:   "LongCat-2.0",
	})

	reply, err := client.Complete(context.Background(), []Message{
		{Role: "system", Content: "be nice"},
		{Role: "user", Content: "ping"},
	})
	require.NoError(t, err)
	assert.Equal(t, "pong", reply)
	assert.Equal(t, "secret", gotAPIKey)
	assert.Equal(t, "Bearer secret", gotAuth)
	assert.Equal(t, anthropicVersion, gotVersion)
	assert.Equal(t, "be nice", gotSystem)
	assert.Equal(t, float64(DefaultMaxTokens), gotMaxTokens)
}

func TestClientCompleteValidation(t *testing.T) {
	client := NewClient(ClientConfig{})

	_, err := client.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model is required")

	client = NewClient(ClientConfig{Model: "m"})
	_, err = client.Complete(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one message")
}
