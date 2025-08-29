// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// Type is documented at https://ai.google.dev/api/caching#Type
//
// The official Go SDK documentation is at https://pkg.go.dev/google.golang.org/genai#Type
type Type string

const (
	TypeUnspecified Type = "TYPE_UNSPECIFIED"
	TypeString      Type = "STRING"
	TypeNumber      Type = "NUMBER"
	TypeInteger     Type = "INTEGER"
	TypeBoolean     Type = "BOOLEAN"
	TypeArray       Type = "ARRAY"
	TypeObject      Type = "OBJECT"
	TypeNull        Type = "NULL"
)

// Format is described at https://spec.openapis.org/oas/v3.0.3#data-types
//
// Values outside of the consts are acceptable.
type Format string

const (
	// For TypeNumber only:
	FormatFloat  Format = "float"
	FormatDouble Format = "double"

	// For TypeInteger only:
	FormatInt32 Format = "int32"
	FormatInt64 Format = "int64"

	// For TypeString only:
	FormatEnum     Format = "enum"
	FormatDate     Format = "date"
	FormatDateTime Format = "date-time"
	FormatByte     Format = "byte"
	FormatPassword Format = "password"
	FormatEmail    Format = "email"
	FormatUUID     Format = "uuid"
)

// Schema is documented at https://ai.google.dev/api/caching#Schema
//
// The official Go SDK documentation is at https://pkg.go.dev/google.golang.org/genai#Schema
//
// Adapted to have less pointers and use omitzero.
type Schema struct {
	// Optional. The value should be validated against any (one or more) of the subschemas
	// in the list.
	AnyOf []*Schema `json:"anyOf,omitzero"`
	// Optional. Default value of the data.
	Default any `json:"default,omitzero"`
	// Optional. The description of the data.
	Description string `json:"description,omitzero"`
	// Optional. Possible values of the element of primitive type with enum format. Examples:
	// 1. We can define direction as : {type:STRING, format:enum, enum:["EAST", NORTH",
	// "SOUTH", "WEST"]} 2. We can define apartment number as : {type:INTEGER, format:enum,
	// enum:["101", "201", "301"]}
	Enum []string `json:"enum,omitzero"`
	// Optional. Example of the object. Will only populated when the object is the root.
	Example any `json:"example,omitzero"`
	// Optional. The format of the data. Supported formats: for NUMBER type: "float", "double"
	// for INTEGER type: "int32", "int64" for STRING type: "email", "byte", etc
	Format Format `json:"format,omitzero"`
	// Optional. SCHEMA FIELDS FOR TYPE ARRAY Schema of the elements of Type.ARRAY.
	Items *Schema `json:"items,omitzero"`
	// Optional. Maximum number of the elements for Type.ARRAY.
	MaxItems int64 `json:"maxItems,omitzero,string"`
	// Optional. Maximum length of the Type.STRING
	MaxLength int64 `json:"maxLength,omitzero,string"`
	// Optional. Maximum number of the properties for Type.OBJECT.
	MaxProperties int64 `json:"maxProperties,omitzero,string"`
	// Optional. Maximum value of the Type.INTEGER and Type.NUMBER
	Maximum float64 `json:"maximum,omitzero"`
	// Optional. Minimum number of the elements for Type.ARRAY.
	MinItems int64 `json:"minItems,omitzero,string"`
	// Optional. SCHEMA FIELDS FOR TYPE STRING Minimum length of the Type.STRING
	MinLength int64 `json:"minLength,omitzero,string"`
	// Optional. Minimum number of the properties for Type.OBJECT.
	MinProperties int64 `json:"minProperties,omitzero,string"`
	// Optional. Minimum value of the Type.INTEGER and Type.NUMBER.
	Minimum float64 `json:"minimum,omitzero"`
	// Optional. Indicates if the value may be null.
	Nullable bool `json:"nullable,omitzero"`
	// Optional. Pattern of the Type.STRING to restrict a string to a regular expression.
	Pattern string `json:"pattern,omitzero"`
	// Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of Type.OBJECT.
	Properties map[string]Schema `json:"properties,omitzero"`
	// Optional. The order of the properties. Not a standard field in open API spec. Only
	// used to support the order of the properties.
	PropertyOrdering []string `json:"propertyOrdering,omitzero"`
	// Optional. Required properties of Type.OBJECT.
	Required []string `json:"required,omitzero"`
	// Optional. The title of the Schema.
	Title string `json:"title,omitzero"`
	// Optional. The type of the data.
	Type Type `json:"type,omitzero"`
}

// FromGoObj generates a gemini.Schema from a Go object.
func (s *Schema) FromGoObj(v any) error {
	val := reflect.ValueOf(v)
	if val.Kind() == reflect.Pointer {
		val = val.Elem()
	}
	t := val.Type()
	return s.FromGoType(t, "", t.Name())
}

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
		fallthrough
	default:
		return fmt.Errorf("unsupported type: %s", t.Kind())
	}
	return nil
}

func convertValue(s string, kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return i, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return u, nil
	case reflect.Float32, reflect.Float64:
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		return f, nil
	case reflect.Bool:
		b, err := strconv.ParseBool(s)
		if err != nil {
			return nil, err
		}
		return b, nil
	case reflect.String:
		return s, nil
	case reflect.Invalid, reflect.Uintptr, reflect.Complex64, reflect.Complex128, reflect.Array, reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.Struct, reflect.UnsafePointer:
		fallthrough
	default:
		return nil, fmt.Errorf("failed to convert example value %v for type %s", s, kind)
	}
}
