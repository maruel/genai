# Goal

Implement exerciseGen and exerciseGenDoc in scoreboard/scoreboard.go.

# Plan

- Read the file DEAR_LLM.md to understand the project
- Take inspiration from tests in internal/internaltest/scoreboard.go
- Instead of testing a predefined hard-coded scenario, exercise the provider and supported models to figure
  out what is supported.
- Important: you only have access to the provider cerebras, no other provider is accessible. You only can use
  cerebras to implement a test. Do not use any other provider.
- Look at the current scoreboard in providers/cerebras/client.go for inspiration of what should be expected.
- Take your time.
- Plan and take notes in a file PROGRESS.md.
- Do one small logical task at a time and update PROGRESS.md.
- Do a git commit after each logical step.
- Continue until you are done. If you are stuck, ask for help.
- When the test fails due to an HTTP recording, you can run the program with RECORD=1 to update the recordings
  in scoreboard/testdata/.
