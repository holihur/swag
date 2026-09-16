// Package llm provides a small, dependency free client for OpenAI compatible
// chat completion APIs together with helpers to translate an OpenAPI document
// into multiple languages.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultBaseURL is the default endpoint used when a provider is not specified.
const DefaultBaseURL = "https://api.openai.com/v1"

// DefaultTimeout is the default request timeout.
const DefaultTimeout = 60 * time.Second

// DefaultMaxTokens is the default completion budget for Anthropic style APIs.
const DefaultMaxTokens = 4096

// Supported chat completion protocols.
const (
	// ProtocolOpenAI speaks the OpenAI /chat/completions format.
	ProtocolOpenAI = "openai"
	// ProtocolAnthropic speaks the Anthropic /v1/messages format.
	ProtocolAnthropic = "anthropic"
)

// anthropicVersion is the API version header required by Anthropic style APIs.
const anthropicVersion = "2023-06-01"

// Message is a single chat message sent to an OpenAI compatible API.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatClient is the minimal interface required to translate documents.
// It is intentionally small so it can be easily mocked in tests.
type ChatClient interface {
	// Complete sends the given messages and returns the assistant reply.
	Complete(ctx context.Context, messages []Message) (string, error)
}

// ClientConfig configures a Client.
type ClientConfig struct {
	// BaseURL is the API root, e.g. https://api.openai.com/v1.
	// It may also point directly at a chat endpoint.
	BaseURL string

	// APIKey is used to authenticate requests when not empty.
	APIKey string

	// Model is the model name used for chat completions.
	Model string

	// Protocol selects the wire format: ProtocolOpenAI or ProtocolAnthropic.
	// When empty it is inferred from BaseURL (URLs containing "anthropic"
	// use the Anthropic protocol).
	Protocol string

	// MaxTokens is the completion budget for Anthropic style APIs.
	// Defaults to DefaultMaxTokens.
	MaxTokens int

	// Timeout is the request timeout. Defaults to DefaultTimeout.
	Timeout time.Duration

	// HTTPClient optionally overrides the underlying HTTP client.
	HTTPClient *http.Client

	// Headers are additional headers added to every request.
	Headers map[string]string
}

// Client is a minimal OpenAI/Anthropic compatible chat completion client.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	protocol   string
	maxTokens  int
	httpClient *http.Client
	headers    map[string]string
}

// compile time assertion that Client implements ChatClient.
var _ ChatClient = (*Client)(nil)

// NewClient creates a new chat completion client.
func NewClient(cfg ClientConfig) *Client {
	protocol := strings.ToLower(strings.TrimSpace(cfg.Protocol))
	if protocol == "" {
		if strings.Contains(strings.ToLower(cfg.BaseURL), "anthropic") {
			protocol = ProtocolAnthropic
		} else {
			protocol = ProtocolOpenAI
		}
	}

	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		if protocol == ProtocolAnthropic {
			baseURL = "https://api.anthropic.com"
		} else {
			baseURL = DefaultBaseURL
		}
	}
	baseURL = strings.TrimRight(baseURL, "/")
	baseURL = endpointURL(baseURL, protocol)

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}

	return &Client{
		baseURL:    baseURL,
		apiKey:     cfg.APIKey,
		model:      cfg.Model,
		protocol:   protocol,
		maxTokens:  maxTokens,
		httpClient: httpClient,
		headers:    cfg.Headers,
	}
}

// endpointURL appends the protocol specific path to a base URL unless it is
// already present.
func endpointURL(baseURL, protocol string) string {
	switch protocol {
	case ProtocolAnthropic:
		if strings.HasSuffix(baseURL, "/v1/messages") || strings.HasSuffix(baseURL, "/messages") {
			return baseURL
		}
		return baseURL + "/v1/messages"
	default:
		if strings.HasSuffix(baseURL, "/chat/completions") {
			return baseURL
		}
		return baseURL + "/chat/completions"
	}
}

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *apiError `json:"error"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Error *apiError `json:"error"`
}

// Complete implements ChatClient.
func (c *Client) Complete(ctx context.Context, messages []Message) (string, error) {
	if c.model == "" {
		return "", fmt.Errorf("llm: model is required")
	}
	if len(messages) == 0 {
		return "", fmt.Errorf("llm: at least one message is required")
	}

	if c.protocol == ProtocolAnthropic {
		return c.completeAnthropic(ctx, messages)
	}
	return c.completeOpenAI(ctx, messages)
}

func (c *Client) completeOpenAI(ctx context.Context, messages []Message) (string, error) {
	body, err := json.Marshal(chatRequest{
		Model:       c.model,
		Messages:    messages,
		Temperature: 0,
	})
	if err != nil {
		return "", fmt.Errorf("llm: marshal request: %w", err)
	}

	data, status, err := c.do(ctx, body, ProtocolOpenAI)
	if err != nil {
		return "", err
	}
	if status < 200 || status >= 300 {
		return "", fmt.Errorf("llm: unexpected status %d: %s", status, strings.TrimSpace(string(data)))
	}

	var parsed chatResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", fmt.Errorf("llm: decode response: %w", err)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return "", fmt.Errorf("llm: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("llm: response contains no choices")
	}

	return parsed.Choices[0].Message.Content, nil
}

func (c *Client) completeAnthropic(ctx context.Context, messages []Message) (string, error) {
	request := anthropicRequest{
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Temperature: 0,
	}

	for _, message := range messages {
		if message.Role == "system" {
			if request.System != "" {
				request.System += "\n\n"
			}
			request.System += message.Content
			continue
		}
		request.Messages = append(request.Messages, anthropicMessage{Role: message.Role, Content: message.Content})
	}

	if len(request.Messages) == 0 {
		return "", fmt.Errorf("llm: at least one non-system message is required")
	}

	body, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("llm: marshal request: %w", err)
	}

	data, status, err := c.do(ctx, body, ProtocolAnthropic)
	if err != nil {
		return "", err
	}
	if status < 200 || status >= 300 {
		return "", fmt.Errorf("llm: unexpected status %d: %s", status, strings.TrimSpace(string(data)))
	}

	var parsed anthropicResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", fmt.Errorf("llm: decode response: %w", err)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return "", fmt.Errorf("llm: %s", parsed.Error.Message)
	}

	var builder strings.Builder
	for _, block := range parsed.Content {
		if block.Type == "text" {
			builder.WriteString(block.Text)
		}
	}
	if builder.Len() == 0 {
		return "", fmt.Errorf("llm: response contains no text content")
	}

	return builder.String(), nil
}

// do sends the request and returns the response body and status code.
func (c *Client) do(ctx context.Context, body []byte, protocol string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, 0, fmt.Errorf("llm: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		if protocol == ProtocolAnthropic {
			// Some Anthropic compatible providers (e.g. LongCat) expect a
			// bearer token, while Anthropic itself uses x-api-key. Send both.
			req.Header.Set("x-api-key", c.apiKey)
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
			req.Header.Set("anthropic-version", anthropicVersion)
		} else {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}
	}
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("llm: request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("llm: read response: %w", err)
	}

	return data, resp.StatusCode, nil
}
