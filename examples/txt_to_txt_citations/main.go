// Send entire documents and leverage providers which support automatic citations
// (Cohere, Anthropic) to leverage their functionality for a supercharged RAG.
//
// This requires `COHERE_API_KEY` (https://dashboard.cohere.com/api-keys)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/cohere"
)

func main() {
	ctx := context.Background()
	c, err := cohere.New(ctx, &genai.ProviderOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Source: On the Origin of Species by Charles Darwin
	// https://www.vliz.be/docs/Zeecijfers/Origin_of_Species.pdf
	const context = `
When on board H.M.S. 'Beagle,' as naturalist, I was much struck with certain facts in the
distribution of the inhabitants of South America, and in the geological relations of the present to the
past inhabitants of that continent. These facts seemed to me to throw some light on the origin of
species--that mystery of mysteries, as it has been called by one of our greatest philosophers. On
my return home, it occurred to me, in 1837, that something might perhaps be made out on this
question by patiently accumulating and reflecting on all sorts of facts which could possibly have
any bearing on it. After five years' work I allowed myself to speculate on the subject, and drew up
some short notes; these I enlarged in 1844 into a sketch of the conclusions, which then seemed to
me probable: from that period to the present day I have steadily pursued the same object. I hope
that I may be excused for entering on these personal details, as I give them to show that I have not
been hasty in coming to a decision.
`
	msgs := genai.Messages{{
		Requests: []genai.Request{
			{Doc: genai.Doc{Filename: "On-the-Origin-of-Species-by-Charles-Darwin.txt", Src: strings.NewReader(context)}},
			{Text: "When did Darwin arrive home?"},
		},
	}}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range res.Replies {
		for _, ci := range r.Citations {
			fmt.Printf("Citation: %s\n", ci.Text)
		}
	}
	fmt.Printf("\nAnswer: %s\n", res.String())
}
