// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import (
	"bytes"
	"context"
	"crypto/tls"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"time"
)

type Retryable struct {
	Transport  http.RoundTripper
	RetryCount int
}

func (t *Retryable) RoundTrip(req *http.Request) (*http.Response, error) {
	var in []byte
	if req.Body != nil && req.GetBody == nil {
		var err error
		if in, err = io.ReadAll(req.Body); err != nil {
			return nil, err
		}
		req.Body = io.NopCloser(bytes.NewBuffer(in))
	}
	resp, err := t.Transport.RoundTrip(req)
	ctx := req.Context()
	for retry := 0; retry < t.RetryCount && shouldRetry(ctx, err, resp); retry++ {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if req.Body != nil {
			if req.GetBody != nil {
				var err2 error
				if req.Body, err2 = req.GetBody(); err2 != nil {
					break
				}
			} else {
				req.Body = io.NopCloser(bytes.NewBuffer(in))
			}
		}
		timer := time.NewTimer(backoff(retry, resp))
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
		resp, err = t.Transport.RoundTrip(req)
	}
	return resp, err
}

// Unwrap implements roundtrippers.Unwrapper.
func (t *Retryable) Unwrap() http.RoundTripper {
	return t.Transport
}

//

var (
	// redirectsErrorRe matches the error returned by net/http when the configured number of redirects is
	// exhausted. This error isn't typed specifically so we resort to matching on the error string.
	redirectsErrorRe = regexp.MustCompile(`stopped after \d+ redirects\z`)
	// schemeErrorRe matches the error returned by net/http when the scheme specified in the URL is invalid.
	// This error isn't typed specifically so we resort to matching on the error string.
	schemeErrorRe = regexp.MustCompile(`unsupported protocol scheme`)
	// invalidHeaderErrorRe matches the error returned by net/http when a request header or value is invalid.
	// This error isn't typed specifically so we resort to matching on the error string.
	invalidHeaderErrorRe = regexp.MustCompile(`invalid header`)
	// notTrustedErrorRe matches the error returned by net/http when the TLS certificate is not trusted. This
	// error isn't typed specifically so we resort to matching on the error string.
	notTrustedErrorRe = regexp.MustCompile(`certificate is not trusted`)
)

func shouldRetry(ctx context.Context, err error, resp *http.Response) bool {
	if ctx.Err() != nil {
		return false
	}
	if err != nil {
		if v, ok := err.(*url.Error); ok {
			if _, ok := v.Err.(*tls.CertificateVerificationError); ok {
				return false
			}
			if s := v.Error(); redirectsErrorRe.MatchString(s) || schemeErrorRe.MatchString(s) || invalidHeaderErrorRe.MatchString(s) || notTrustedErrorRe.MatchString(s) {
				return false
			}
		}
		return true
	}
	return resp.StatusCode == http.StatusTooManyRequests || // 429
		resp.StatusCode == http.StatusBadGateway ||
		resp.StatusCode == http.StatusServiceUnavailable ||
		resp.StatusCode == http.StatusGatewayTimeout
}

func parseRetryAfterHeader(header string) (time.Duration, bool) {
	if sleep, err := strconv.ParseInt(header, 10, 64); err == nil {
		if sleep > 0 {
			return time.Second * time.Duration(sleep), true
		}
	} else if retryTime, err := time.Parse(time.RFC1123, header); err == nil {
		if until := time.Until(retryTime); until > 0 {
			return until, true
		}
	}
	return 0, false
}

func backoff(retries int, resp *http.Response) time.Duration {
	// "Retry-After" is generally sent along HTTP 429.
	if sleep, ok := parseRetryAfterHeader(resp.Header.Get("Retry-After")); ok {
		return sleep
	}
	return time.Duration(math.Pow(2, float64(retries))) * time.Second
}
