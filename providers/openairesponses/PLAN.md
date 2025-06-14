# Project overview

Implement providers/openairesponses/client.go, taking inspiration from providers/openai/client.go.

## Resources

The difference is that providers/openai/client.go uses the old openai chat completion API. The new package providers/openairesponses/client.go uses the new openai responses API.

I saved the API description from https://platform.openai.com/docs/api-reference/responses into the following files:
- openai_responses_create.txt
- openai_responses_get.txt
- openai_responses_list.txt
- openai_responses_object.txt
- openai_responses_streaming.txt

No need to use the web.

## Plan

1. Start by scaffolding client.go. Take inspiration from provider/openai/client.go. Try to keep similar order
   and style.
1. Implement Client struct and func New first.
1. Start implementing genai.ProviderGen.
    1. Implement GenSync first, GenStream will be done later.
1. Implement the necessary structures as described in openai_responses_create.txt and openai_responses_object.txt.
1. Leverage openai_responses_get.txt and openai_responses_object.txt
1. Stop there. The user will review the implementation, course correct and then the next steps will be taken.
1. When the user asks you to continue work, implement GenStream using knowledge from openai_responses_streaming.txt

## Workflow

- Iterate over the plan, creating one commit every time a logical step inside a plan step is done.
- Do not run tests, you do not have access to the API key necessary to run the smoke tests. Make sure the code compiles.
- Leave breadcrumbs for yourself in providers/openairesponses/PLAN_STATUS.md so that you don't have to keep
  everything in your context.
