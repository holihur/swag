package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/go-openapi/spec"
)

// DefaultBatchSize is the maximum number of segments translated per request.
const DefaultBatchSize = 50

// DefaultMaxTextLength is the maximum number of characters sent per request.
const DefaultMaxTextLength = 12000

// translatableKeys are the OpenAPI fields whose values are translated.
// Keys are matched case sensitively as they appear in the OpenAPI document.
var translatableKeys = map[string]struct{}{
	"title":       {},
	"summary":     {},
	"description": {},
}

// Segment is a single translatable string found in an OpenAPI document.
type Segment struct {
	// Pointer is the RFC 6901 JSON pointer locating the value.
	Pointer string `json:"pointer"`
	// Text is the original value.
	Text string `json:"text"`
}

// TranslateOptions configures how a document is translated.
type TranslateOptions struct {
	// BatchSize is the maximum number of segments per request.
	BatchSize int
	// MaxTextLength is the maximum number of characters per request.
	MaxTextLength int
	// SystemPrompt optionally overrides the default system prompt.
	SystemPrompt string
	// Glossary maps source terms to their preferred translation. It is added
	// to the system prompt so the model keeps terminology consistent.
	Glossary map[string]string
	// Cache, when set, is consulted before calling the LLM and updated with
	// every new translation. It enables incremental translation: unchanged
	// strings are never sent to the LLM again.
	Cache Cache
}

func (o TranslateOptions) batchSize() int {
	if o.BatchSize > 0 {
		return o.BatchSize
	}
	return DefaultBatchSize
}

func (o TranslateOptions) maxTextLength() int {
	if o.MaxTextLength > 0 {
		return o.MaxTextLength
	}
	return DefaultMaxTextLength
}

// TranslateSpec returns a copy of the given swagger document with all
// human readable fields translated into the target language.
func TranslateSpec(ctx context.Context, client ChatClient, swagger *spec.Swagger, language string, opts TranslateOptions) (*spec.Swagger, error) {
	if client == nil {
		return nil, fmt.Errorf("llm: chat client is required")
	}
	if strings.TrimSpace(language) == "" {
		return nil, fmt.Errorf("llm: target language is required")
	}

	raw, err := json.Marshal(swagger)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal swagger: %w", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("llm: unmarshal swagger: %w", err)
	}

	segments := ExtractSegments(doc)
	if len(segments) == 0 {
		return cloneSwagger(swagger)
	}

	translations, err := TranslateSegments(ctx, client, segments, language, opts)
	if err != nil {
		return nil, err
	}

	for pointer, translated := range translations {
		if strings.TrimSpace(translated) == "" {
			continue
		}
		if err := setByPointer(doc, pointer, translated); err != nil {
			return nil, fmt.Errorf("llm: apply translation for %q: %w", pointer, err)
		}
	}

	translatedRaw, err := json.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal translated swagger: %w", err)
	}

	var result spec.Swagger
	if err := json.Unmarshal(translatedRaw, &result); err != nil {
		return nil, fmt.Errorf("llm: unmarshal translated swagger: %w", err)
	}

	return &result, nil
}

func cloneSwagger(swagger *spec.Swagger) (*spec.Swagger, error) {
	raw, err := json.Marshal(swagger)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal swagger: %w", err)
	}
	var result spec.Swagger
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("llm: unmarshal swagger: %w", err)
	}
	return &result, nil
}

// ExtractSegments walks the decoded OpenAPI document and returns every
// translatable string together with its JSON pointer.
func ExtractSegments(doc any) []Segment {
	var segments []Segment
	walk(doc, "", &segments)
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Pointer < segments[j].Pointer
	})
	return segments
}

func walk(node any, pointer string, segments *[]Segment) {
	switch value := node.(type) {
	case map[string]any:
		for key, child := range value {
			childPointer := pointer + "/" + escapePointer(key)
			if text, ok := child.(string); ok {
				if _, ok := translatableKeys[key]; ok && strings.TrimSpace(text) != "" {
					*segments = append(*segments, Segment{Pointer: childPointer, Text: text})
				}
				continue
			}
			walk(child, childPointer, segments)
		}
	case []any:
		for i, child := range value {
			walk(child, pointer+"/"+strconv.Itoa(i), segments)
		}
	}
}

// TranslateSegments translates the given segments in batches.
// It returns a map from JSON pointer to the translated text.
//
// When opts.Cache is set, previously translated strings are reused and only
// new or changed strings are sent to the LLM, making repeated runs incremental.
func TranslateSegments(ctx context.Context, client ChatClient, segments []Segment, language string, opts TranslateOptions) (map[string]string, error) {
	result := make(map[string]string, len(segments))

	fingerprint := opts.fingerprint()

	pending := make([]Segment, 0, len(segments))
	keys := make(map[string]string, len(segments))

	for _, segment := range segments {
		key := CacheKey(language, fingerprint, segment.Text)
		if opts.Cache != nil {
			if cached, ok := opts.Cache.Get(key); ok {
				result[segment.Pointer] = cached
				continue
			}
		}
		pending = append(pending, segment)
		keys[segment.Pointer] = key
	}

	if len(pending) == 0 {
		return result, nil
	}

	for _, batch := range batchSegments(pending, opts.batchSize(), opts.maxTextLength()) {
		translated, err := translateBatch(ctx, client, batch, language, opts)
		if err != nil {
			return nil, err
		}
		for pointer, text := range translated {
			result[pointer] = text
			if opts.Cache != nil {
				if key, ok := keys[pointer]; ok {
					opts.Cache.Set(key, text)
				}
			}
		}
	}

	return result, nil
}

// CacheKey derives the cache key for a source string. It includes the target
// language and a fingerprint of the translation options so that changing the
// prompt or glossary invalidates previously cached translations.
func CacheKey(language, fingerprint, source string) string {
	sum := sha256.Sum256([]byte(language + "\x00" + fingerprint + "\x00" + source))
	return language + ":" + hex.EncodeToString(sum[:])
}

// fingerprint returns a stable digest of the options that influence the
// translation output.
func (o TranslateOptions) fingerprint() string {
	hash := sha256.New()
	hash.Write([]byte(o.SystemPrompt))

	terms := make([]string, 0, len(o.Glossary))
	for source := range o.Glossary {
		terms = append(terms, source)
	}
	sort.Strings(terms)
	for _, source := range terms {
		hash.Write([]byte("\x00"))
		hash.Write([]byte(source))
		hash.Write([]byte("\x00"))
		hash.Write([]byte(o.Glossary[source]))
	}

	return hex.EncodeToString(hash.Sum(nil))
}

func batchSegments(segments []Segment, batchSize, maxTextLength int) [][]Segment {
	var batches [][]Segment

	var current []Segment
	currentLength := 0

	for _, segment := range segments {
		segmentLength := len(segment.Text)
		if len(current) > 0 && (len(current) >= batchSize || currentLength+segmentLength > maxTextLength) {
			batches = append(batches, current)
			current = nil
			currentLength = 0
		}
		current = append(current, segment)
		currentLength += segmentLength
	}

	if len(current) > 0 {
		batches = append(batches, current)
	}

	return batches
}

func translateBatch(ctx context.Context, client ChatClient, segments []Segment, language string, opts TranslateOptions) (map[string]string, error) {
	payload := make(map[string]string, len(segments))
	for _, segment := range segments {
		payload[segment.Pointer] = segment.Text
	}

	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal translation payload: %w", err)
	}

	messages := []Message{
		{Role: "system", Content: systemPrompt(language, opts)},
		{Role: "user", Content: "Translate the string values of this JSON object:\n" + string(encoded)},
	}

	response, err := client.Complete(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("llm: translate into %q: %w", language, err)
	}

	translated, err := parseTranslationResponse(response)
	if err != nil {
		return nil, fmt.Errorf("llm: parse translation into %q: %w", language, err)
	}

	return translated, nil
}

func systemPrompt(language string, opts TranslateOptions) string {
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		return fmt.Sprintf(opts.SystemPrompt, language)
	}

	prompt := fmt.Sprintf(`You are a professional technical translator for OpenAPI documentation.
Translate every JSON value into %s.
Strict rules:
- Keep the JSON keys exactly as they are.
- Return a single valid JSON object with the same keys.
- Preserve Markdown formatting, placeholders such as {id} or %s, URLs, code snippets and proper nouns.
- Do not translate identifiers, enum values or example code.
- Output only the JSON object, without explanations or code fences.`, language, "%s")

	if len(opts.Glossary) > 0 {
		terms := make([]string, 0, len(opts.Glossary))
		for source := range opts.Glossary {
			terms = append(terms, source)
		}
		sort.Strings(terms)

		var builder strings.Builder
		builder.WriteString("\nUse the following glossary, mapping the source term to the preferred translation:\n")
		for _, source := range terms {
			fmt.Fprintf(&builder, "- %s => %s\n", source, opts.Glossary[source])
		}
		prompt += builder.String()
	}

	return prompt
}

func parseTranslationResponse(raw string) (map[string]string, error) {
	raw = strings.TrimSpace(raw)
	raw = stripCodeFence(raw)

	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start == -1 || end == -1 || end < start {
		return nil, fmt.Errorf("response does not contain a JSON object")
	}

	var decoded map[string]any
	if err := json.Unmarshal([]byte(raw[start:end+1]), &decoded); err != nil {
		return nil, err
	}

	result := make(map[string]string, len(decoded))
	for key, value := range decoded {
		if text, ok := value.(string); ok {
			result[key] = text
		}
	}

	return result, nil
}

func stripCodeFence(raw string) string {
	if !strings.HasPrefix(raw, "```") {
		return raw
	}

	raw = strings.TrimPrefix(raw, "```")
	if idx := strings.Index(raw, "\n"); idx != -1 {
		// Drop an optional language hint such as "json".
		if !strings.Contains(raw[:idx], "{") {
			raw = raw[idx+1:]
		}
	}
	raw = strings.TrimSuffix(strings.TrimSpace(raw), "```")
	return raw
}

// escapePointer escapes a JSON pointer reference token (RFC 6901).
func escapePointer(token string) string {
	token = strings.ReplaceAll(token, "~", "~0")
	return strings.ReplaceAll(token, "/", "~1")
}

func unescapePointer(token string) string {
	token = strings.ReplaceAll(token, "~1", "/")
	return strings.ReplaceAll(token, "~0", "~")
}

// setByPointer sets the value located by the given JSON pointer.
func setByPointer(root any, pointer string, value string) error {
	tokens := splitPointer(pointer)
	if len(tokens) == 0 {
		return fmt.Errorf("empty pointer")
	}

	var current any = root
	for i, token := range tokens {
		last := i == len(tokens)-1
		switch node := current.(type) {
		case map[string]any:
			next, ok := node[token]
			if !ok {
				return fmt.Errorf("key %q not found", token)
			}
			if last {
				node[token] = value
				return nil
			}
			current = next
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil {
				return fmt.Errorf("invalid array index %q", token)
			}
			if index < 0 || index >= len(node) {
				return fmt.Errorf("array index %d out of range", index)
			}
			if last {
				node[index] = value
				return nil
			}
			current = node[index]
		default:
			return fmt.Errorf("cannot descend into %T at %q", current, token)
		}
	}

	return nil
}

func splitPointer(pointer string) []string {
	pointer = strings.TrimPrefix(pointer, "/")
	if pointer == "" {
		return nil
	}
	parts := strings.Split(pointer, "/")
	for i, part := range parts {
		parts[i] = unescapePointer(part)
	}
	return parts
}
