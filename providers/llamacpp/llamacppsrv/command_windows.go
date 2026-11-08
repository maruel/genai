// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Windows subprocess creation without a console window.

package llamacppsrv

import (
	"os/exec"
	"syscall"
)

func hideConsole(cmd *exec.Cmd) {
	// CREATE_NO_WINDOW prevents console allocation even when the host has no console.
	// Explicit stdout/stderr handles still carry server logs and probe output.
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createNoWindow
}
