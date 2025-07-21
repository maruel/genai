# Goal

Implement exerciseGen and exerciseGenDoc in scoreboard/scoreboard.go to exercise the provider and dynamically
discover the supported functionalities of the provider.

# Plan

- Read the file DEAR_LLM.md to understand the project
- Take inspiration from tests in internal/internaltest/scoreboard.go. The goal here is NOT to write tests in
  scoreboard.go. Instead of testing a predefined hard-coded scenario, exercise the provider and supported
  models to figure out what is supported at runtime and populate a fresh new Scenario instance.
- Important: you only have access to the provider cerebras, no other provider is accessible. You only can use
  cerebras to implement a test. Do not use any other provider.
- Look at the current scoreboard in providers/cerebras/client.go for inspiration of what should be expected.
- Plan and take notes in a file PROGRESS.md.
- Do one small logical task at a time and update PROGRESS.md.
- MODIFY ONLY the following files: PROGRESS.md, scoreboard/scoreboard.go, and scoreboard/scoreboard_test.go
- Run the command "git add . && git commit -a -m Checkpoint" after each logical step.
- Continue until you are done. If you are stuck, ask for help.

# Testing

- Use the command `go test -timeout 1m ./scoreboard`
- When the test fails due to an HTTP recording, you can run the program with the environment variable
  `RECORD=1` set to update the recordings in scoreboard/testdata/.
