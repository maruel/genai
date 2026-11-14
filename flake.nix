# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

{
  description = "genai: a Go client library for large language models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    # The repository's Python tooling is one ruff dependency from uv.lock.
    # Building that environment with uv2nix instead of letting `uv run` do it
    # keeps the wheel patched for NixOS, where uv's manylinux download cannot
    # find a dynamic loader. It also makes the environment available offline.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      # nixpkgs 26.11 dropped x86_64-darwin; the remaining platforms match what
      # the Go toolchain and uv.lock's wheels support.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # genai itself is a library; the command-line tools under cmd/ are what
      # this package installs. They are built from the same tree and share its
      # dependencies, so one derivation carries all four.
      genai-tools =
        pkgs:
        pkgs.buildGoModule {
          pname = "genai-tools";
          # There is no separate release artifact for the tools; the checkout
          # they come from identifies the build.
          version = self.shortRev or "dirty";
          src = self;

          subPackages = [
            "cmd/cache-mgr"
            "cmd/list-models"
            "cmd/llama-serve"
            "cmd/scoreboard"
          ];

          vendorHash = "sha256-sThRPflOEQFfCBhwq8gT16sKOd3MpPg9D/LOP/IXcRs=";

          # The provider tests replay recorded HTTP sessions, but the suite also
          # expects live credentials for the recording-only paths. `make test`
          # is the repository's entry point for them; the package build stays
          # offline and fast.
          doCheck = false;

          meta = {
            description = "Command-line tools for the genai library";
            homepage = "https://github.com/maruel/genai";
            license = lib.licenses.asl20;
            platforms = lib.platforms.unix;
          };
        };

      # The uv workspace is system-independent; the overlay only needs the
      # locked wheel set. Every package then comes from a fixed-output
      # derivation, so `uv run --locked ruff` and the Python unittest step in
      # `make verify` do not reach the network.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      pythonOverlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      pythonSets = forAllSystems (
        pkgs:
        (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; }).overrideScope
          (lib.composeManyExtensions [ pythonOverlay ])
      );
    in
    {
      packages = forAllSystems (pkgs: {
        default = genai-tools pkgs;
        genai-tools = genai-tools pkgs;
      });

      # Handy for `nix flake check`: the derivation builds every command,
      # including the packages they import, without the test suite.
      checks = forAllSystems (pkgs: {
        default = genai-tools pkgs;
      });

      devShells = forAllSystems (
        pkgs:
        let
          pythonSet = pythonSets.${pkgs.stdenv.hostPlatform.system};
          virtualenv = pythonSet.mkVirtualEnv "genai-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              # go.mod pins 1.27.3, and GOTOOLCHAIN below refuses to fetch a
              # different one.
              pkgs.go_1_27
              pkgs.gopls

              # The ruff environment and the `uv run` that make verify calls.
              virtualenv
              pkgs.uv

              pkgs.git
            ];

            env = {
              # uv must not manage a second environment: the one from
              # mkVirtualEnv already has the locked ruff, patched for NixOS.
              UV_NO_SYNC = "1";
              UV_PROJECT_ENVIRONMENT = "${virtualenv}";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";

              # A go.mod bump should fail here instead of silently downloading
              # another Go release.
              GOTOOLCHAIN = "local";
            };

            shellHook = ''
              # Nixpkgs' Python builders leak PYTHONPATH into unrelated
              # environments; drop it so uv and the unittest step use the
              # virtualenv.
              unset PYTHONPATH
            '';
          };
        }
      );
    };
}
