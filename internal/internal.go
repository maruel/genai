// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/maruel/httpjson"
)

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

// ClientBase implements the shared HTTP client functionality used across all API clients.
type ClientBase[E any] struct {
	// ClientJSON is exported for testing replay purposes.
	ClientJSON httpjson.Client
	// APIKeyURL is the URL to present to the user upon authentication error.
	APIKeyURL string
}

// DoRequest performs an HTTP request and handles error responses.
//
// It takes care of sending the request, decoding the response, and handling errors.
// All API clients should use this method for their HTTP communication needs.
func (c *ClientBase[E]) DoRequest(ctx context.Context, method, url string, in, out any) error {
	resp, err := c.ClientJSON.Request(ctx, method, url, nil, in)
	if err != nil {
		return err
	}
	var er E
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				// Check if the error message already contains an API key URL
				errorMsg := fmt.Sprintf("%v", &er)
				if !strings.Contains(errorMsg, "API key") || !strings.Contains(errorMsg, "http") {
					return fmt.Errorf("%w: %v. You can get a new API key at %s", herr, errorMsg, c.APIKeyURL)
				}
				return fmt.Errorf("%w: %s", herr, errorMsg)
			}
			return fmt.Errorf("%w: %v", herr, &er)
		}
		return fmt.Errorf("%v", &er)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, http.StatusText(herr.StatusCode), c.APIKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, http.StatusText(herr.StatusCode))
		}
		return err
	}
}

// DecodeError handles HTTP error responses from API calls.
//
// It handles JSON decoding of error responses and provides appropriate error messages
// with context such as API key URLs for unauthorized errors.
func (c *ClientBase[E]) DecodeError(ctx context.Context, url string, resp *http.Response) error {
	var er E
	switch i, err := httpjson.DecodeResponse(resp, &er); i {
	case 0:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %v. You can get a new API key at %s", herr, &er, c.APIKeyURL)
			}
			return fmt.Errorf("%w: %v", herr, &er)
		}
		return fmt.Errorf("%v", &er)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, http.StatusText(herr.StatusCode), c.APIKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, http.StatusText(herr.StatusCode))
		}
		return err
	}
}
