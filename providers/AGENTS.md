When investigating a feature or doing gap analysis compared to an official SDK, first do a git clone to
investigate more efficiently. Only then look at the online documentation to confirm if needed.

When recording smoketests against live APIs, run fewer models at a time (use `-run`) to avoid exceeding
the default 10-minute `go test` timeout. Reasoning models in particular can generate tens of thousands
of tokens per test, making a full recording run take 20+ minutes.
