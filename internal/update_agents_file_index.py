#!/usr/bin/env python3
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Update AGENTS.md files (containing a file index marker) with an auto-generated index.

To opt-in a directory, add these two markers to its AGENTS.md:

    <!-- BEGIN FILE INDEX -->
    <!-- END FILE INDEX -->

The script auto-discovers all AGENTS.md files tracked by git that contain the
markers, generates a file index from first-line comments, and injects it between
the markers. It also ensures a CLAUDE.md symlink exists next to every AGENTS.md.
"""

import fnmatch
import os
import re
import subprocess
import sys


def get_git_files():
    try:
        result = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True)
        return [f for f in result.stdout.split("\0") if f]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error listing git files: {e}", file=sys.stderr)
        return []


def _skip_frontmatter(lines):
    """Return the index of the first line after any YAML front-matter, or 0 if none.

    Handles --- delimited front-matter and bare YAML (no delimiters).
    """
    if not lines or not lines[0]:
        return 0
    first = lines[0].strip()
    if first == "---":
        # Standard delimited front-matter: skip until closing ---.
        for i, line in enumerate(lines[1:], 1):
            if not line or line.strip() == "---":
                return i + 1
        return len(lines)
    if re.match(r"^[a-z_]+\s*:", first):
        # Bare YAML: ends at first blank line.
        for i, line in enumerate(lines):
            if not line or not line.strip():
                return i
        return len(lines)
    return 0


def _py_docstring(lines, i):
    """Extract the description from a Python triple-quoted docstring starting at lines[i].

    Returns the description string, or "" if none found.
    """
    sline = lines[i].strip()
    quote = sline[:3]
    # Single-line docstring: """text"""
    if sline.endswith(quote) and len(sline) > 6:
        return sline[3:-3].strip()
    # Multi-line: return the first content line.
    content = sline[3:].strip()
    if content:
        return content
    # Opening quotes on their own line; use next non-empty line.
    for j in range(i + 1, len(lines)):
        if lines[j] and lines[j].strip():
            return lines[j].strip()
    return ""


def get_file_description(filepath):
    """Return the description for a file, or None if not applicable.

    Returns None if the file type has no comment convention (or is explicitly
    excluded). Returns "" if the file supports comments but has no description,
    which is treated as an error by callers.
    """
    # Glob patterns mapping filenames to their comment prefix. None skips the file.
    comment_prefixes = {
        "*.d.ts": None,
        "pnpm-lock.yaml": None,
        "*.cjs": "//",
        "*.go": "//",
        "*.js": "//",
        "*.kt": "//",
        "*.md": "#",
        "*.mjs": "//",
        "*.py": "#",
        "*.sh": "#",
        "*.swift": "//",
        "*.ts": "//",
        "*.tsx": "//",
        "*.yaml": "#",
        "*.yml": "#",
        "Dockerfile*": "#",
        "Makefile": "#",
    }
    if os.path.islink(filepath):
        return None
    fname = os.path.basename(filepath)
    match = next(((pat, p) for pat, p in comment_prefixes.items() if fnmatch.fnmatch(fname, pat)), None)
    if match is None:
        return None  # extension not recognised
    _, prefix = match
    if not prefix:
        return None  # explicitly excluded pattern
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(20)]
    if fname.endswith(".md"):
        # Skip front-matter; first # heading after it is the description.
        start_idx = _skip_frontmatter(lines)
    else:
        start_idx = 1 if (lines[0] and lines[0].startswith("#!")) else 0
    # If the first content line is a copyright header, skip lines until a blank.
    for i in range(start_idx, len(lines)):
        if not lines[i]:
            return ""
        if not lines[i].strip():
            continue
        if " copyright " in lines[i].lower():
            # Skip the entire copyright block (all lines up to the next blank).
            i += 1
            while i < len(lines) and lines[i] and lines[i].strip():
                i += 1
            start_idx = i
        break
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if not line:
            break
        sline = line.strip()
        if not sline:
            continue
        if fname.endswith(".py") and (sline.startswith('"""') or sline.startswith("'''")):
            return _py_docstring(lines, i)
        # Skip common directives/metadata that aren't descriptions.
        if sline.startswith(f"{prefix}go:"):
            continue
        if sline.startswith(f"{prefix} +build"):
            continue
        if sline.startswith(f"{prefix} nolint"):
            continue
        if sline.startswith(f"{prefix} swift-tools-version:"):
            continue
        if sline.startswith(prefix):
            comment = sline[len(prefix) :].strip()
            if not comment:
                continue
            return comment
        # Hit code before a comment.
        return ""
    return ""


def discover_configs(all_files):
    """Auto-discover workspace roots from AGENTS.md files that contain a file index marker.

    Returns a dict mapping target AGENTS.md path to its set of excluded child directories.
    """
    candidates = sorted(f for f in all_files if os.path.basename(f) == "AGENTS.md")
    configs = {"AGENTS.md": set()}
    for f in candidates:
        with open(f, "r", encoding="utf-8") as fh:
            if "<!-- BEGIN FILE INDEX -->" in fh.read():
                configs[f] = set()
    # For each config, find child workspaces and add them to exclude_dirs.
    for target, exclude in configs.items():
        root = os.path.dirname(target)
        prefix = root + "/" if root else ""
        for other_target in configs:
            oroot = os.path.dirname(other_target)
            if oroot == root:
                continue
            if not prefix:
                child_rel = oroot
            elif oroot.startswith(prefix):
                child_rel = oroot[len(prefix) :]
            else:
                continue
            if "/" not in child_rel:
                exclude.add(child_rel)
    return configs


def generate_index(target, exclude, all_files, all_configs):
    """Generate the file index for target, returning (content, missing) where
    missing is a list of files that support comments but have no description."""
    root_dir = os.path.dirname(target)
    files_found = []
    missing = []
    for filepath in all_files:
        # Skip own AGENTS.md.
        if filepath == target:
            continue
        # Scope to root_dir.
        if root_dir:
            if not filepath.startswith(root_dir + "/"):
                continue
            relpath = filepath[len(root_dir) + 1 :]
        else:
            relpath = filepath
        # Check excluded subdirectories, but let sub-workspace AGENTS.md through.
        rel_parts = relpath.replace("\\", "/").split("/")
        if rel_parts[0] in exclude:
            if filepath not in all_configs:
                continue
        # Skip any file in a testdata/ directory.
        if "testdata" in rel_parts:
            continue
        desc = get_file_description(filepath)
        if desc is None:
            continue  # file type has no comment convention
        if desc == "":
            missing.append(relpath)
        else:
            files_found.append((relpath, desc))
    desc = "Autogenerated from first-line comments. Run scripts/update_agents_file_index.py to refresh."
    lines = ["## File Index", "", desc, ""]
    for path, comment in sorted(files_found):
        lines.append(f"- `{path}`: {comment}")
    return "\n".join(lines), missing


def update_markdown(target_file, content):
    if not os.path.exists(target_file):
        print(f"Warning: {target_file} not found, skipping.")
        return
    start = "<!-- BEGIN FILE INDEX -->"
    end = "<!-- END FILE INDEX -->"
    with open(target_file, "r", encoding="utf-8") as f:
        original = f.read()
    new_section = f"{start}\n{content}\n{end}"
    if start in original and end in original:
        pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
        updated = pattern.sub(new_section, original)
    else:
        updated = (original.rstrip() + "\n\n" + new_section + "\n") if original.strip() else (new_section + "\n")
    if updated == original:
        return
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Updated: {target_file}")


def ensure_claude_symlinks(all_files):
    """Ensure every AGENTS.md has a sibling CLAUDE.md symlink pointing to it."""
    for f in all_files:
        if os.path.basename(f) != "AGENTS.md":
            continue
        d = os.path.dirname(f) or "."
        link = os.path.join(d, "CLAUDE.md")
        if os.path.islink(link) and os.readlink(link) == "AGENTS.md":
            continue
        if os.path.exists(link):
            print(f"Error: {link} exists but is not a symlink to AGENTS.md.", file=sys.stderr)
            return 1
        os.symlink("AGENTS.md", link)
        print(f"Created: {link} -> AGENTS.md")
    return 0


def main():
    all_files = get_git_files()
    if not all_files:
        print("No files found in git repository.")
        return 1
    ret = ensure_claude_symlinks(all_files)
    if ret:
        return ret
    configs = discover_configs(all_files)
    all_missing = []
    for target, exclude in configs.items():
        content, missing = generate_index(target, exclude, all_files, configs)
        update_markdown(target, content)
        all_missing.extend(missing)
    if all_missing:
        print("Error: the following files have no description comment:", file=sys.stderr)
        for f in sorted(all_missing):
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
