// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Gemini provider Zod schema utilities for tool calling.

package gemini

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// FromGoObj generates a gemini.Schema from a Go object.
func (s *Schema) FromGoObj(v any) error {
	val := reflect.ValueOf(v)
	if val.Kind() == reflect.Pointer {
		val = val.Elem()
	}
	t := val.Type()
	return s.FromGoType(t, "", t.Name())
}

// FromGoType converts a Go type to a Schema.
func (s *Schema) FromGoType(t reflect.Type, tag reflect.StructTag, parent string) error {
	if strings.HasPrefix(t.String(), "reflect.") {
		return fmt.Errorf("received a reflect type: %s", t.String())
	}
	// TODO: Add support for:
	// - anchor
	// - anyof_ref
	// - anyof_required
	// - anyof_type
	// - jsonschema_extras
	// - oneof_ref
	// - oneof_required
	// - oneof_type
	// - pattern
	// - format
	// - readOnly
	// - writeOnly
	// - multipleOf
	// - minimum
	// - maximum
	// - exclusiveMaximum
	// - exclusiveMinimum
	// - uniqueItems
	// - and more, see *Keywords() at https://github.com/invopop/jsonschema/blob/main/reflect.go
	jsonschemaTag := tag.Get("jsonschema")
	for part := range strings.SplitSeq(jsonschemaTag, ",") {
		if after, found := strings.CutPrefix(part, "enum="); found {
			s.Enum = append(s.Enum, after)
		} else if after, found := strings.CutPrefix(part, "default="); found {
			if converted, err := convertValue(after, t.Kind()); err == nil {
				s.Default = converted
			} else {
				return err
			}
		} else if after, found := strings.CutPrefix(part, "example="); found {
			if converted, err := convertValue(after, t.Kind()); err == nil {
				s.Example = converted
			} else {
				return err
			}
		} else if after, found := strings.CutPrefix(part, "description="); found {
			s.Description = after
		} else if after, found := strings.CutPrefix(part, "title="); found {
			s.Title = after
		} else if after, found := strings.CutPrefix(part, "minLength="); found {
			if i, err := strconv.ParseInt(after, 10, 64); err == nil {
				s.MinLength = i
			} else {
				return err
			}
		} else if after, found := strings.CutPrefix(part, "maxLength="); found {
			if i, err := strconv.ParseInt(after, 10, 64); err == nil {
				s.MaxLength = i
			} else {
				return err
			}
		} else if after, found := strings.CutPrefix(part, "minItems="); found {
			if i, err := strconv.ParseInt(after, 10, 64); err == nil {
				s.MinItems = i
			} else {
				return err
			}
		} else if after, found := strings.CutPrefix(part, "maxItems="); found {
			if i, err := strconv.ParseInt(after, 10, 64); err == nil {
				s.MaxItems = i
			} else {
				return err
			}
		} else if after, found := strings.CutPrefix(part, "type="); found {
			s.Type = Type(strings.ToUpper(after))
		} else if part != "" {
			return fmt.Errorf("unknown jsonschema tag: %q", part)
		}
	}
	if desc := tag.Get("jsonschema_description"); desc != "" {
		s.Description = desc
	}
	if s.Type != "" {
		return nil
	}
	switch t.Kind() {
	case reflect.String:
		s.Type = TypeString
	case reflect.Bool:
		s.Type = TypeBoolean
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		s.Type = TypeInteger
		if t.Kind() == reflect.Int32 {
			s.Format = FormatInt32
		} else if t.Kind() == reflect.Int64 {
			s.Format = FormatInt64
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		s.Type = TypeInteger
	case reflect.Float32:
		s.Type = TypeNumber
		s.Format = FormatFloat
	case reflect.Float64:
		s.Type = TypeNumber
		s.Format = FormatDouble
	case reflect.Array, reflect.Slice:
		s.Type = TypeArray
		if t.Kind() == reflect.Array {
			s.MinItems = int64(t.Len())
			s.MaxItems = int64(t.Len())
		}
		itemsSchema := &Schema{}
		if err := itemsSchema.FromGoType(t.Elem(), reflect.StructTag(""), parent); err != nil {
			return fmt.Errorf("failed to convert array/slice element type: %w", err)
		}
		s.Items = itemsSchema
	case reflect.Struct:
		if t.PkgPath() == "time" && t.Name() == "Time" {
			s.Type = TypeString
			s.Format = FormatDateTime
			return nil
		}
		s.Type = TypeObject
		s.Properties = make(map[string]Schema)
		if t.NumField() == 0 {
			return nil
		}
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			jsonTag := field.Tag.Get("json")
			jsonName := strings.Split(jsonTag, ",")[0]
			if jsonName == "-" {
				continue
			}
			if jsonName == "" {
				jsonName = field.Name
			}

			propSchema := Schema{}
			p := parent
			if p != "" {
				p += "."
			}
			p += field.Name
			if err := propSchema.FromGoType(field.Type, field.Tag, p); err != nil {
				return fmt.Errorf("failed to convert property %q: %w", field.Name, err)
			}
			s.Properties[jsonName] = propSchema
			if t.NumField() > 1 {
				s.PropertyOrdering = append(s.PropertyOrdering, jsonName)
			}
			if jsonTag == "-" {
				continue
			}
			if !strings.Contains(jsonTag, "omitempty") && !strings.Contains(jsonTag, "omitzero") && field.Type.Kind() != reflect.Pointer {
				s.Required = append(s.Required, jsonName)
			}
		}
	case reflect.Map:
		s.Type = TypeObject
		if k := t.Key().Kind(); k != reflect.String {
			return fmt.Errorf("unsupported map key type %q for schema generation; only string keys are supported", k)
		}
		// For maps, we don't have predefined properties like structs.
		// The Schema struct doesn't have AdditionalProperties to describe arbitrary key-value pairs.
		// So, we set Type to TypeObject and leave Properties empty.
	case reflect.Pointer:
		s.Nullable = true
		return s.FromGoType(t.Elem(), tag, parent) // Pass tag to underlying element.
	case reflect.Invalid, reflect.Complex64, reflect.Complex128, reflect.Chan, reflect.Func, reflect.Interface, reflect.UnsafePointer:
		return fmt.Errorf("unsupported type: %s", t.Kind())
	default:
		return fmt.Errorf("unsupported type: %s", t.Kind())
	}
	return nil
}

func convertValue(s string, kind reflect.Kind) (json.RawMessage, error) {
	switch kind {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return marshalJSONRaw(i)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return marshalJSONRaw(u)
	case reflect.Float32, reflect.Float64:
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		return marshalJSONRaw(f)
	case reflect.Bool:
		b, err := strconv.ParseBool(s)
		if err != nil {
			return nil, err
		}
		return marshalJSONRaw(b)
	case reflect.String:
		return marshalJSONRaw(s)
	case reflect.Invalid, reflect.Uintptr, reflect.Complex64, reflect.Complex128, reflect.Array, reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.Struct, reflect.UnsafePointer:
		return nil, fmt.Errorf("failed to convert example value %v for type %s", s, kind)
	default:
		return nil, fmt.Errorf("failed to convert example value %v for type %s", s, kind)
	}
}

func marshalJSONRaw(v any) (json.RawMessage, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(b), nil
}
