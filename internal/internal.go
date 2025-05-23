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

	"github.com/maruel/httpjson"
)

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

// DecodeError handles HTTP error responses from API calls.
//
// It handles JSON decoding of error responses and provides appropriate error messages
// with context such as API key URLs for unauthorized errors.
func DecodeError(ctx context.Context, url string, resp *http.Response, er fmt.Stringer, apiKeyURL string) error {
	switch i, err := httpjson.DecodeResponse(resp, er); i {
	case 0:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if apiKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, er.String(), apiKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, er.String())
		}
		return errors.New(er.String())
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if apiKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, http.StatusText(herr.StatusCode), apiKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, http.StatusText(herr.StatusCode))
		}
		return err
	}
}
