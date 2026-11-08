// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Subprocess creation on platforms without Windows console allocation.

//go:build !windows

package llamacppsrv

import "os/exec"

func hideConsole(_ *exec.Cmd) {}
