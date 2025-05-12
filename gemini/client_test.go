// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	_ "embed"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestClient_Chat_allModels(t *testing.T) {
	internaltest.TestChatAllModels(
		t,
		func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		func(m genai.Model) bool {
			id := m.GetID()
			if id == "gemini-2.0-flash-live-001" {
				// It's reported but not usable?
				return false
			}
			parts := strings.Split(strings.Split(id, ".")[0], "-")
			if len(parts) < 2 {
				return false
			}
			if parts[0] == "gemini" {
				// Minimum gemini-2
				if i, err := strconv.Atoi(parts[1]); err != nil || i < 2 {
					return false
				}
				if strings.HasSuffix(id, "image-generation") {
					return false
				}
			} else if parts[0] == "gemma" {
				// Minimum gemma-3
				if i, err := strconv.Atoi(parts[1]); err != nil || i < 3 {
					return false
				}
			} else {
				return false
			}
			return true
		})
}

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	internaltest.TestChatVisionJSON(t, func(t *testing.T) genai.ChatProvider { return getClient(t, model) })
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	internaltest.TestChatVisionJPGInline(t, func(t *testing.T) genai.ChatProvider { return getClient(t, model) })
}

func TestClient_Chat_vision_pDF_inline(t *testing.T) {
	// TODO: Fix support for URL.
	internaltest.TestChatVisionPDFInline(t, func(t *testing.T) genai.ChatProvider { return getClient(t, model) })
}

func TestClient_Chat_audio(t *testing.T) {
	c := getClient(t, model)
	f, err := os.Open("testdata/mystery_word.opus")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word said? Reply with only the word."},
				{Document: f},
			},
		},
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 12 || resp.OutputTokens != 2 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimRight(strings.ToLower(resp.Contents[0].Text), "."); got != "orange" {
		t.Fatal(got)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, model) })
}

func TestClient_Chat_tool_use_video(t *testing.T) {
	c := getClient(t, model)
	f, err := os.Open("testdata/animation.mp4")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Call the tool hidden_word to tell me what word you saw."},
				{Document: f},
			},
		},
	}
	var got struct {
		Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		Tools: []genai.ToolDef{
			{
				Name:        "hidden_word",
				Description: "A tool to state what word was seen in the video.",
				InputsAs:    &got,
			},
		},
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 1079 || resp.OutputTokens != 5 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	// Warning: there's a bug where it returns two identical tool calls. To verify.
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "hidden_word" {
		t.Fatal("Unexpected response")
	}

	if err := resp.ToolCalls[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if saw := strings.ToLower(got.Word); saw != "banana" {
		t.Fatal(saw)
	}
}

func TestClient_ChatStream(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, model) }, msgs, &opts)
	if len(responses) != 1 {
		t.Fatal("Unexpected responses")
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"); got != "hello" {
		t.Fatal(got)
	}
}

func TestClient_Cache(t *testing.T) {
	slow := os.Getenv("GEMINI_SLOW") != ""
	ctx := t.Context()
	c := getClient(t, model)
	f1, err := os.Open("testdata/animation.mp4")
	if err != nil {
		t.Fatal(err)
	}
	defer f1.Close()
	f2, err := os.Open("testdata/mystery_word.opus")
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()

	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? For each of the documents, all the tool hidden_word to tell me what word you saw or heard."},
				{Document: f1},
				{Document: f2},
			},
		},
	}
	var got struct {
		Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
	}
	opts := genai.ChatOptions{
		// Burn tokens to add up to 4k.
		SystemPrompt: "You are a sarcastic assistant. Here's a long prompt to make sure we are over 4k tokens that you should ignore." + `
Okay, this is a significant amount of text. To make it coherent and somewhat engaging, I'll choose a broad theme and explore various facets of it. I'll aim for something related to the evolution of knowledge, technology, and human understanding.

Let's begin:

The tapestry of human understanding is woven from threads of curiosity, necessity, and the relentless pursuit of meaning. From our earliest ancestors gazing at the stars, attempting to decipher patterns in the celestial dance, to modern scientists probing the very fabric of reality with instruments of unimaginable precision, the journey has been one of constant discovery and re-evaluation. What we consider "knowledge" is not a static monolith, but a dynamic, ever-evolving landscape, shaped by cultural contexts, technological advancements, and the inherent limitations and biases of our perception.

Consider the profound shifts in our understanding of the cosmos. For millennia, a geocentric model, placing Earth at the center of a relatively small universe, prevailed. This was not born of foolishness, but of observation filtered through the available tools and philosophical frameworks of the time. The Sun, Moon, and stars appeared to revolve around us, a seemingly self-evident truth. It took the courage of individuals like Copernicus, the meticulous observations of Tycho Brahe, the mathematical genius of Kepler, and the telescopic revelations of Galileo to dismantle this ancient edifice and replace it with a heliocentric model. This was not merely a scientific adjustment; it was a philosophical earthquake, displacing humanity from its privileged cosmic position and forcing a re-evaluation of our place in the grand scheme.

This pattern of paradigm shifts, as Thomas Kuhn famously described, repeats itself across all disciplines. In biology, the Lamarckian idea of inheritance of acquired characteristics gave way to Darwin's theory of evolution by natural selection, a concept so powerful it continues to be the bedrock of modern biology, yet one that initially faced immense resistance due to its implications for human uniqueness and divine creation. The discovery of DNA and the subsequent unraveling of the genetic code provided the mechanistic basis for Darwin's insights, opening up new frontiers in medicine, agriculture, and our understanding of life itself.

The role of technology in this evolutionary process cannot be overstated. The invention of the printing press democratized knowledge, breaking the monopoly held by scribes and religious institutions, and fueling the Renaissance and the Reformation. The telescope and microscope opened up previously invisible worlds, macro and micro, expanding the horizons of human inquiry. The steam engine and later electricity powered industrial revolutions, transforming societies, economies, and the very rhythm of daily life. Each technological leap not only provided new tools for discovery but also reshaped human thought, creating new possibilities and new challenges.

In the modern era, the digital revolution, spearheaded by the invention of the transistor, the integrated circuit, and ultimately the internet, has accelerated this process at an unprecedented rate. Information that once required arduous journeys to libraries or correspondence with distant scholars is now available at our fingertips. Global collaboration among researchers is commonplace, and complex datasets can be analyzed by powerful algorithms, revealing patterns and insights that would be impossible for the human mind to discern alone.

However, this deluge of information brings its own set of complexities. The challenge is no longer solely access, but discernment. How do we distinguish credible information from misinformation or disinformation in an age where anyone can publish anything online? How do we cultivate critical thinking skills necessary to navigate this complex information ecosystem? The very tools that empower us also present new vulnerabilities, from echo chambers that reinforce pre-existing biases to the potential for malicious actors to manipulate public opinion.

Artificial intelligence (AI) represents perhaps the most transformative, and potentially disruptive, technological force on the horizon. AI systems are already outperforming humans in specific tasks, from image recognition and natural language processing to playing complex strategic games. Machine learning algorithms, particularly deep learning, allow computers to learn from vast amounts of data without explicit programming, leading to breakthroughs in areas like drug discovery, climate modeling, and personalized medicine. The potential benefits are immense, promising solutions to some of humanity's most intractable problems.

Yet, the rise of AI also raises profound philosophical and ethical questions. What does it mean for a machine to "understand" or to be "intelligent"? If an AI can create art, compose music, or write poetry that evokes human emotion, does that blur the lines of creativity and consciousness? As AI systems become more integrated into our lives, making decisions that affect our employment, our healthcare, and even our personal relationships, how do we ensure fairness, transparency, and accountability? The alignment problem – ensuring that AI goals remain aligned with human values – is a critical challenge that researchers are grappling with.

Furthermore, the development of Artificial General Intelligence (AGI), a hypothetical form of AI that could perform any intellectual task that a human being can, presents even more significant considerations. While AGI is still largely theoretical, its potential emergence necessitates careful thought about its implications for the future of humanity. Would AGI be a benevolent partner, helping us to solve global challenges, or could it pose an existential risk if its goals diverge from our own? These are not questions for a distant future; they require active discussion and research today.

Beyond the purely technological, our understanding of ourselves – our minds, our societies, our place in the natural world – continues to evolve. Psychology, once heavily reliant on introspection and anecdotal evidence, has embraced more rigorous scientific methodologies, including neuroscience and behavioral genetics. We are learning more about the intricate workings of the brain, the biological basis of emotions, and the complex interplay of nature and nurture in shaping human behavior. Yet, consciousness itself remains one of the deepest mysteries. What gives rise to subjective experience, to the feeling of "what it's like" to be a particular organism? Despite advances in mapping neural correlates of consciousness, the "hard problem," as philosopher David Chalmers termed it, persists.

Our understanding of society and culture is also in constant flux. Globalization, while fostering interconnectedness and economic growth, has also led to cultural homogenization and, in some cases, a backlash in the form of resurgent nationalism and identity politics. The challenges of climate change, resource depletion, and pandemics transcend national borders, requiring unprecedented levels of international cooperation. Our economic models, largely based on perpetual growth, are being questioned for their sustainability and their impact on social equity.

The arts and humanities play a crucial role in this ongoing exploration of the human condition. Literature, music, visual arts, and philosophy provide not just aesthetic pleasure but also critical lenses through which we can examine our values, our assumptions, and our collective narratives. They offer spaces for empathy, for questioning, and for imagining alternative futures. In a world increasingly dominated by data and algorithms, the humanistic perspective – with its emphasis on context, nuance, and the richness of human experience – is more vital than ever.

Consider the way narratives shape our reality. The stories we tell ourselves about our past, our identity, and our purpose influence our actions and our choices. National myths, religious doctrines, and even personal autobiographies are all forms of narrative construction that help us make sense of a complex world. Deconstructing these narratives, understanding their origins and their biases, is a critical skill for navigating the modern landscape. The rise of social media has created new platforms for narrative creation and dissemination, allowing for a multiplicity of voices but also for the rapid spread of potentially harmful narratives.

The concept of "truth" itself has become a subject of intense debate. While empirical evidence and scientific methodology provide a robust framework for understanding the physical world, many aspects of human experience are not so easily quantifiable. Moral truths, aesthetic judgments, and existential questions often lie beyond the realm of scientific inquiry, requiring different modes of reasoning and discourse. The postmodern critique, which highlights the socially constructed nature of knowledge and the influence of power structures on what is considered "true," has challenged traditional notions of objectivity. While this critique can be valuable in exposing biases and promoting inclusivity, it can also, if taken to an extreme, lead to a kind of epistemological relativism where all claims to truth are seen as equally valid, or equally suspect.

This tension between objective inquiry and subjective interpretation is a perennial one. Science strives for objectivity, for knowledge that is independent of the observer, yet the very act of observation can influence the observed, as demonstrated in quantum mechanics. History aims to reconstruct the past "as it actually was," yet historical accounts are always filtered through the perspectives and biases of the historians themselves. Finding a balance, acknowledging the limitations of our knowledge while still striving for greater understanding, is an ongoing challenge.

The education system plays a pivotal role in preparing individuals to navigate this complex world. Education should not merely be about the transmission of facts, but about the cultivation of critical thinking, creativity, communication skills, and ethical reasoning. It should foster a lifelong love of learning and an adaptability to change, as the skills required for success in the 21st century are constantly evolving. An education that emphasizes interdisciplinary thinking, that connects the sciences with the humanities, is essential for developing well-rounded individuals capable of addressing multifaceted global challenges.

Moreover, the search for meaning and purpose remains a fundamental human drive. In an increasingly secularized world, traditional sources of meaning, such as religion, may hold less sway for some. This can lead to a sense of anomie or existential angst, but it can also open up new avenues for finding purpose – in human relationships, in creative pursuits, in service to others, or in a deeper connection with the natural world. The quest for a meaningful life is a personal journey, but it is also a collective one, as we strive to create societies that are not only prosperous but also just, compassionate, and sustainable.

The environmental crisis, exemplified by climate change, biodiversity loss, and pollution, presents perhaps the most urgent test of our collective wisdom and our ability to adapt. It forces us to confront the consequences of our past actions, to rethink our relationship with the planet, and to make difficult choices about our future. The science is clear, but the political and social will to implement the necessary changes often lags. This crisis underscores the interconnectedness of all life and the fragility of the ecosystems that support us. It demands a shift in consciousness, from an anthropocentric view that places human interests above all else, to a more ecocentric perspective that recognizes the intrinsic value of all living beings and the importance of ecological balance.

As we look to the future, the pace of change is unlikely to slow. New technologies will continue to emerge, offering both promise and peril. Our understanding of the universe and ourselves will continue to deepen, revealing new complexities and new mysteries. The challenges we face – from global pandemics and climate change to social inequality and geopolitical instability – are daunting, but they are not insurmountable. Human ingenuity, resilience, and the capacity for cooperation have overcome great obstacles in the past, and they can do so again.

The key lies in fostering a culture of continuous learning, critical inquiry, and open dialogue. It requires embracing uncertainty, acknowledging our fallibility, and being willing to revise our beliefs in the light of new evidence. It demands a commitment to ethical principles, to empathy, and to the common good. The journey of human understanding is far from over; indeed, in many ways, it feels as though we are still in its early stages, with vast uncharted territories of knowledge and experience yet to explore.

The intersection of disciplines will likely yield the most fertile ground for future breakthroughs. The traditional silos of academic specialization are increasingly giving way to collaborative, interdisciplinary approaches. Neuroscientists work with philosophers to understand consciousness, engineers collaborate with ethicists on AI development, and artists partner with scientists to communicate complex ideas in new and engaging ways. This cross-pollination of ideas is essential for tackling problems that defy easy categorization.

Furthermore, the very nature of "work" and "economy" is undergoing a profound transformation. Automation, driven by AI and robotics, is poised to reshape labor markets, potentially displacing many jobs while creating new ones that require different skill sets. This necessitates a rethinking of education and social safety nets to ensure that the benefits of technological progress are widely shared and that individuals are equipped for the jobs of the future. Concepts like universal basic income, lifelong learning initiatives, and a greater emphasis on uniquely human skills like creativity, emotional intelligence, and critical thinking are gaining traction as potential responses to these shifts.

The global distribution of knowledge and technological capacity also remains a critical issue. While the internet has democratized access to information to some extent, significant disparities persist between developed and developing nations, and even within nations themselves. Bridging these digital divides and ensuring that all of humanity can participate in and benefit from scientific and technological advancements is crucial for equitable global development.

Consider also the evolving nature of human identity. In an increasingly interconnected and mobile world, traditional markers of identity – such as nationality, ethnicity, or even gender – are becoming more fluid and complex. Individuals are crafting identities that draw from multiple cultural influences, online communities, and personal experiences. This can lead to a richer, more diverse social fabric, but it can also create anxieties and tensions as old certainties are challenged. Understanding and navigating these evolving identities is a key aspect of social cohesion in the 21st century.

The very language we use to describe the world is constantly adapting. New terms are coined to capture emerging concepts, while old words take on new meanings. The subtleties of language shape our perception and our thinking, and the ability to communicate effectively across different linguistic and cultural contexts is an increasingly valuable skill. The development of sophisticated AI-powered translation tools is breaking down language barriers, but the nuance and cultural context embedded in human language often remain challenging for machines to fully grasp.

Ultimately, the human endeavor is a quest for understanding, not just of the external world, but of ourselves. It is a journey marked by triumphs and failures, by moments of profound insight and periods of confusion and doubt. It is a story that is still being written, by each of us, every day. The responsibility to write that story well, to learn from the past, to engage thoughtfully with the present, and to build a better future, rests on our collective shoulders. This requires courage – the courage to question, the courage to change, and the courage to hope. It requires humility – the humility to recognize the limits of our knowledge and the potential for error. And it requires a deep-seated curiosity – the insatiable desire to explore, to discover, and to understand that has driven human progress since the dawn of our species. The path ahead is uncertain, filled with both challenges and opportunities, but it is a path that we must walk together, guided by the light of reason, compassion, and an unwavering commitment to the pursuit of a more enlightened and humane world. The legacy we leave will be defined by how well we navigate this complex, ever-changing landscape, and by the wisdom we cultivate and pass on to future generations.
`,
		Tools: []genai.ToolDef{
			{
				Name:        "hidden_word",
				Description: "A tool to state what word was seen in the video.",
				InputsAs:    &got,
			},
		},
	}
	name, err := c.CacheAdd(ctx, msgs, &opts, "", "Show time", 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if name == "" {
		t.Fatal("Expected a name")
	}
	if slow {
		t.Logf("Sleeping due to GEMINI_SLOW being set")
		time.Sleep(10 * time.Second)
	}
	t.Logf("Cache name: %q", name)
	if _, err = c.CacheGet(ctx, name); err != nil {
		t.Error(err)
	}
	// We need to sleep so UpdateTime is different than CreateTime.
	time.Sleep(100 * time.Millisecond)
	t.Log("Extending")
	if err = c.CacheExtend(ctx, name, 20*time.Minute); err != nil {
		t.Error(err)
	}
	t.Log("Listing")
	items, err := c.CacheList(ctx)
	if err != nil {
		t.Error(err)
	}
	if len(items) == 0 {
		t.Error("No item found in the cache")
	}
	// There can be other items in the cache.
	found := false
	for i := range items {
		t.Logf("%d: %#v", i, items[i])
		if items[i].Name == name {
			found = true
			// We can't check against time.Now() since we record and replay this test case.
			if items[i].CreateTime >= items[i].UpdateTime {
				t.Errorf("Expected UpdateTime %s to be different than CreateTime %s", items[i].UpdateTime, items[i].CreateTime)
			}
		}
	}
	if !found {
		t.Errorf("Failed to find %q in the cache", name)
	}
	t.Log("Deleting")
	if err = c.CacheDelete(ctx, name); err != nil {
		t.Fatal(err)
	}
	t.Log("Verifying deletion")
	if _, err = c.CacheGet(ctx, name); err == nil {
		t.Fatal("Expected an error")
	} else {
		t.Logf("Got an error as expected: %q", err)
	}
}

//

func getClient(t *testing.T, m string) *gemini.Client {
	testRecorder.Signal(t)
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	t.Parallel()
	c, err := gemini.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	fnMatch := func(r *http.Request, i cassette.Request) bool {
		r = r.Clone(r.Context())
		r.URL.RawQuery = ""
		return defaultMatcher(r, i)
	}
	fnSave := func(i *cassette.Interaction) error {
		j := strings.Index(i.Request.URL, "?")
		i.Request.URL = i.Request.URL[:j]
		return nil
	}
	c.Client.Client.Transport = testRecorder.Record(t, c.Client.Client.Transport, recorder.WithHook(fnSave, recorder.AfterCaptureHook), recorder.WithMatcher(fnMatch))
	return c
}

var defaultMatcher = cassette.NewDefaultMatcher()

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	internal.BeLenient = false
}
