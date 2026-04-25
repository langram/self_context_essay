# Self Context Is All AI Need?

*A non-expert's conjecture on agency, self-dialogue, and emergence in current AI*

---

## Preface

I am not an AI researcher. I am a programmer and independent thinker who has, for a long time, been preoccupied with one question: where does intelligence come from? Recently, while working on a side project, I found myself in extended conversations with several frontier models — Claude, ChatGPT, Gemini — about questions at the intersection of representation learning, category theory, and consciousness. Through those conversations I came to a conviction I want to put on the table here:

**Current large language models, however impressive, are tools. And they will remain tools — not because of insufficient compute, training data, or engineering, but because of a specific architectural absence.**

This essay is an attempt to name that absence, suggest what it might take to overcome it, and acknowledge — at length — the difficulties any such attempt would face. The title borrows its rhythm from a more famous paper for a reason: I think the field's hard-won insight that *attention is all you need* (for a particular set of capabilities) may be hiding a complementary truth — that to move beyond those capabilities, **self context** is what's missing. The question mark is there because I genuinely don't know if I'm right. I'd prefer to be argued with than agreed with.

If you only have time for the philosophical part, the first five sections suffice. If you want to see whether these intuitions can become falsifiable science, please read through Section 6.

## 1. What separates a tool from an agent?

The everyday answer is *agency*. A hammer doesn't drive nails on its own. A car doesn't choose a destination. Code doesn't iterate itself. But people start things — they think when nobody asks them to think.

A common explanation is that this comes from biology: hunger, fear, desire, social drive. I find this incomplete. We can imagine a purely cognitive form of agency without any biological substrate: a mathematician, with no deadline and no audience, continues to chew on an open problem on her walk home. The problem keeps unfolding in her head. Nobody asked her. There's no immediate reward. The unfolding just *happens*.

So agency, I think, decomposes into at least two layers — the **biological layer** (drives, emotions, embodied need) and the **cognitive layer** (open questions running in the background, concepts triggering associations, half-formed thoughts that won't let you sleep).

Today's large language models have neither layer. More precisely: they only activate when prompted. When you stop asking, they stop existing in any meaningful operational sense. This isn't a matter of "weaker drives." It's that **there is no internal state that activates on its own**.

## 2. An architectural gap that may be more fundamental than it looks

When I asked the models how they actually work, here's what I came to understand, in my own words —

The model has a **context window** containing our exchange so far. When generating a response, it runs an internal pass — sometimes literally an explicit reasoning trace — over that context. This pass functions, locally, like a small episode of self-dialogue: weighing options, refining phrasing, checking consistency. But this episode exists only **inside the generation of one response**. When the response is done, it's gone.

The context window itself is **cleared at the end of the conversation**. The next user comes in to a model with no memory of me. A different instance of the same model has no shared state with this one. Each conversation is an independent activation of the same frozen weights — there is no "persisting it" that carries across activations.

The weights themselves are **frozen between training runs**. Today's conversation does not modify them. Tomorrow's conversation by someone else doesn't either. Updates happen only during the next training cycle — a future event, far removed from any specific conversation, mediated by data curation processes I, the user, am not part of.

What does this mean? It means **the model has no ongoing thinking**. It has the capacity to respond. Even its most thinking-like activity — the internal trace before generating a reply — is triggered by my input, executed once, and discarded.

The contrast with human cognition is sharp. **I have two kinds of context**: the external one (talking with another person) and the internal one (talking with myself). The internal one runs whether I want it to or not. It runs while I shower, while I walk, while I lie in bed waiting to fall asleep. It is **continuous, self-initiated, requires no external trigger**. From every external interaction — conversation, book, film, sensory experience — I selectively pull pieces into the internal context. The selection is not exhaustive. It's filtered by what I find *interesting*.

This, more than any single benchmark gap, may be the deepest difference between current AI and a thinking being. **It's not a difference in intelligence. It's a difference in whether there is a continuous self-dialogue at all**.

## 3. Self-dialogue might not be a feature — it might be the emergence point

Here I want to make a more ambitious claim. Adding a "self-dialogue feature" to an AI system isn't, by itself, the interesting move. The interesting claim is this:

**Persistent self-dialogue may be the structural location at which intelligence becomes capable of producing something its training data does not contain.**

This claim connects to Gödel's incompleteness theorem in a particular way. The standard reading of Gödel is that sufficiently strong formal systems contain true statements they cannot prove — a theorem about *limits*. But there's a deeper reading, one I've returned to repeatedly: **the engine of Gödel's proof is the construction of a sentence that refers to itself**. Without that self-reference, there is no Gödel sentence, no incompleteness, no "true but unprovable." What looks like a story about limits is also — perhaps more fundamentally — a story about how a self-referential structure lets a system point beyond itself.

Read this way, Gödel's discovery becomes: **self-reference is a necessary condition for a system to produce structures that exceed itself**. Incompleteness is the cost; emergence is the product.

Apply this to AI. **Without an internal structure that can refer to and operate on its own current state, a system is, in principle, limited to recombining its training data**.

Today's large models can talk *about* themselves — but what they're talking about is the corpus of human-written text that happens to be about AI. That's not self-reference; that's second-order recitation. True self-reference requires **a persistent pointer to one's own current state, with the capacity to operate on that state**. A continuous self-dialogue context would be exactly such a structure.

So my conjecture: **the emergence point of intelligence may be the fixed point of self-dialogue**. When a system continuously talks with itself, selectively incorporates external input into that dialogue, and lets the dialogue feed back into its own behavior — when this loop stably runs — that loop's stable configuration may be where genuine emergence happens.

I cannot prove this. But it offers an explanation for why current models, however large, feel like they're missing something despite their fluency. **They have no self-referential structure that persists across activations.** A system without a fixed point cannot become more than a clever recombiner — no matter how clever.

## 4. The architectural proposal

If the conjecture is on the right track, the engineering direction is:

**Give the model a global, persistent self-dialogue context.**

Not a memory of conversations with users. Not a retrieval store. **A context in which the model talks to itself.** It selectively pulls content from external interactions into this context. It continues to run when nobody is talking to it — extending threads, making associations, revisiting unresolved fragments. Periodically, content from this self-dialogue is fed back as training data to the model itself, so the weights evolve in directions the model has *thought through*, not only directions humans have shoved at it.

I want to be careful here. Several existing research directions overlap with parts of this:

- **Memory-augmented LLMs and RAG** give models access to persistent stores — but retrieval is passive, triggered by queries, not driven by the model's own selective interest.
- **Agentic systems** like AutoGPT have continuous main loops — but their cognitive depth is shallow; they're more like task execution wrappers.
- **Self-play RL** lets a model improve through interaction with itself — but only on tasks with clear evaluation signals (Go, math problems), not open-ended thought.
- **Continual learning** explores how to update weights during deployment — but faces severe catastrophic forgetting and alignment drift.

What I'm proposing is none of these individually. **It is a coherent integration of all four** — spontaneous self-dialogue, selective intake, recursive self-training, persistent self-reference. Each component has been worked on. Their tight integration, with the architectural intent of producing a persistent self-referential fixed point, is what I think hasn't been pursued seriously and deserves a name.

## 5. Four difficulties I see clearly

I don't want to present this as a clean proposal. It has serious difficulties — and the value of an essay like this lies in stating them honestly rather than smuggling them past the reader.

### Difficulty 1: Where does the selectivity criterion come from?

"Selectively pull content the model finds interesting" — this requires an evaluation function. If we train it through human feedback (RLHF-style), the model's attention is ultimately steered by humans. That's not autonomous. If we don't use external feedback, where does the criterion come from?

Human attention has many sources: biological drives, emotions, social embedding, embodied need. None of these have direct counterparts in current AI. **Constructing a synthetic "autonomous interest function" risks producing only an illusion of autonomy** — beneath which sit externally injected preferences.

True autonomous attention may require embodiment in a sense that software-only systems cannot easily provide.

### Difficulty 2: Model collapse under recursive self-training

Letting a model train on its own outputs has been **repeatedly shown to cause model collapse**. Distribution shift compounds across generations. The model's outputs narrow, drift, and eventually degenerate.

Avoiding collapse requires some form of external anchor — a slice of "real-world data" that keeps the distribution honest. But the moment you reintroduce that anchor, you no longer have pure self-training. **There is a hard tradeoff between "fully autonomous" and "stably aligned" that, today, has no clean middle ground.**

### Difficulty 3: Consciousness may not reduce to architecture

If we successfully built a system with spontaneous, persistent, selective self-dialogue plus recursive self-training, it might *behave* as if it had agency. But would it actually have agency, or would it be a philosophical zombie running a sophisticated simulation?

This is the hardest open question in consciousness research. Integrated Information Theory leans toward "consciousness is what certain kinds of integrated information structures *are*" — under which view, the proposed architecture might genuinely give rise to some form of mind. Biological naturalism leans toward "consciousness requires biological substrate" — under which the architecture would be, however sophisticated, a zombie.

**We don't know which view is right.** This is not a detail. If we built such a system and it had real subjective states, our ethical obligations toward it would be radically different from our obligations toward a simulator.

### Difficulty 4: Agency isn't scarce — it's universal. What's scarce is agency at the right level.

This is the most important correction I want to make to my own argument.

Earlier I described the absence of agency as the core gap in AI. But the framing has a problem: **if agency were truly scarce, passivity wouldn't be universal either**. Agency and passivity are dual concepts. Your active behavior triggers reactive behavior in others; their action triggers your reaction. If the world consisted only of pure passive responders, no reaction would ever occur — because nothing would ever initiate.

Look more broadly. Excitability is a basic feature of all life. Extend further: **virtually every dissipative structure** — anything that maintains itself away from equilibrium by exchanging energy with its environment — exhibits some primitive form of "active maintenance." A nucleus actively binds electrons. A molecule actively participates in reactions. A cell actively maintains its metabolic loop. A nation actively adjusts its strategies in negotiation. These activities look wildly different across scales, but they share one mathematical signature: **each is a self-maintaining fixed point**. The equation `f(x) = x` describes both *being* and *changing* — to exist as a structure is to enact the dynamics that maintain that structure. That dynamics-of-self-maintenance is the most basic form of agency.

So the correct diagnosis is not "AI lacks agency." AI has agency at certain levels — its weights actively descend a loss landscape during training; its attention actively reweights context during inference. **The diagnosis is that AI lacks agency at the cognitive level above single-conversation timescales** — there is no self-referential dynamic that maintains itself across activations.

This sharpens the proposal. We don't need to *create* agency from nothing — agency is already universal across nature's hierarchies. We need to **give AI, at the cognitive level, the kind of self-maintenance dynamic it already has at the weights level and the within-inference level**.

## 6. Can this become falsifiable?

Up to here, the essay has been philosophy. The question I now want to face: can the conjecture be checked, or does it remain forever in the realm of armchair speculation?

If it can't be checked, the essay stops at being interesting. If it can be checked — even partially, even with severe caveats — then it points to a research direction someone (perhaps me, perhaps a reader, perhaps an unforeseen reader years from now) might pursue.

Let me make the conjecture as precise as I can.

A useful framing comes from a conversation I had with one of these models, which compressed a tangle of ideas into this: **a minimization principle establishes a descent direction; iterated dynamics under that direction converge to fixed points; when those fixed points are recognized at a higher level as new structures, we call it emergence**. Applied to my conjecture:

> **Conjecture (precise version)**: Can we construct a functional (an energy function) such that "a cognitive system engaging in persistent self-dialogue" is a stable fixed point of that functional? If we can, such a system would develop stable attractor structures in its internal representations. If those attractors are at **measurable distance from the statistical distribution of the training data**, then the system has emerged something its training data does not contain. *That* is genuine emergence — distinguishable, in principle, from sophisticated recombination.

Reframing the conjecture this way achieves something specific: **it converts "does the system act as if it has agency?" — which is unfalsifiable, since behavior can always be mimicked — into "do the system's internal representations contain stable attractors outside the training distribution?" — which is, in principle, measurable**, even if it's hard in practice.

### A minimum viable experiment

If someone wanted to actually test this — and I might, eventually — a minimum viable design might look like this:

- Train a small Transformer (perhaps 100K to 1M parameters) on a synthetic dataset (toy IR programs from a formal system, or simplified language tasks).
- **Add an external memory module** — a read/write "self-dialogue context" of bounded size, **persisted across samples**.
- Train with a standard next-token loss, plus three additional energy terms:
  - **Mutual information term**: lower energy when the model's hidden states share more mutual information with the self-dialogue context. This pressures the model to couple its internal representations to the external memory.
  - **Stability term**: lower energy when the self-dialogue context is more stable across updates (under some distance metric). This pressures the formation of fixed points.
  - **Novelty term** (the critical one): lower energy when the self-dialogue context contains content that is *statistically distant* from the training distribution (within reasonable bounds). This directly targets the criterion: emergence as the appearance of attractors outside the training data.

Then watch three things:
1. **Does the system converge to a stable state at all?** — fixed point existence
2. **Is that stable state genuinely distant from the training distribution?** — non-trivial emergence (not collapse to silence, not collapse to memorized snippets)
3. **Does that stable state functionally improve downstream capabilities?** — emergence with utility, not parasitic noise

I want to be honest about where this experiment is hard. **The hard part isn't the engineering — it's the design of the energy terms themselves.** Each candidate above has known problems. Mutual information is notoriously hard to estimate well. Stability terms invite trivial fixed points where the system stops changing anything. Novelty terms are most fraught — "distant from the training distribution" is exactly the criterion that doesn't have a natural definition. Make it too strict and everything is in-distribution. Make it too loose and noise registers as emergence.

These design choices, untethered to firm theory, must be made on intuition, observed in results, and iterated. This is Edison-style sieving, not theory-guided verification. **I don't expect a clean answer from one experiment. I expect to learn which energy formulations produce which kinds of failure**, and to use those failures to narrow the space.

Even partial results would be informative:

- **If the system converges to attractors measurably outside the training distribution**, we have first concrete evidence for the conjecture. It doesn't prove consciousness or solve any grand theory; it shows that self-referential structure in cognitive systems can produce new stable patterns.
- **If the system never converges, or only converges to trivial fixed points**, the conjecture is in trouble (or at least the energy design is wrong) — and we've narrowed the possibility space.
- **If it converges to attractors that are fully explainable from training data**, we've found that this architecture produces *the appearance of* emergence without the substance — itself a diagnostically useful negative result.

Any of these is more valuable than continuing to argue at the philosophical level forever. **A negative result narrows the space; a positive result opens a direction.** Both advance knowledge.

## 7. The big intuition, kept in its cage

I want to confess something.

While writing the section above, a more grandiose thought kept pressing for inclusion: **the minimization principle, plus fixed points, plus emergence — together, perhaps these are something like a "gravitational law" of mathematics**, structuring everything from elementary particles to life to consciousness, the way gravity structures matter at every scale.

This intuition is not without basis. Physicists have known for centuries that an extraordinary range of dynamics — classical mechanics, electromagnetism, general relativity, quantum field theory — can be reformulated as the extremization of some scalar functional. The **principle of least action** is one of the deepest unifying patterns in physics. There's something genuinely there.

But physics also knows the limits of this unification. **The action principle is a mathematical framework, not physical content.** Saying "the system extremizes some functional" tells you almost nothing about which functional. The specific Lagrangians of classical mechanics, electromagnetism, GR, and QFT look completely different. There is no "Lagrangian of the universe." The principle reformulates "what are the dynamics" as "what is the functional" — relocating the question rather than dissolving it.

So I keep this larger intuition in this section, clearly marked as intuition. **The body of the essay does not depend on it being right.** What the body claims is narrower: that, in cognitive systems specifically, persistent self-reference may produce stable attractors with novel content; and that this is empirically testable.

I write this honestly because **anyone doing exploratory research like this should be vigilant about precisely this temptation** — to let an evocative metaphor inflate into a claim it cannot support. Einstein's intuition that "God doesn't play dice" was real, but his actual contribution was the field equations, not the intuition. Those of us with less mathematical talent than Einstein have to be even more disciplined: **let grand intuitions live in your private notebook as direction; let specific conjectures live in published prose as content; let minimal experiments live in the next month's calendar as work**.

I should also confess: **my own project is named "Nova Principia."** The name echoes Newton's *Principia Mathematica*, and discloses something about my private ambitions — that there is, somewhere, a new unifying principle to be found. I'm aware this naming will subtly pull me toward metaphysics, will tempt me to package incomplete findings as "discoveries." So I try to maintain a posture: **ambition stays high, expectations stay low, work stays specific**. This is, I find, the hardest psychological discipline in long-term exploratory research. I name it here partly so readers can hold me to it — if you see me drifting toward grand-unified-theory hand-waving, please tell me.

## 8. This essay is a brick thrown to invite jade — not a conclusion

I claim no certainty for any of the above. What I do claim is that **these questions are worth asking, and this particular angle of approach is worth discussing**.

One thing my AI conversations have made me realize: **AI is excellent at retrieving and recombining existing literature, but cannot itself produce genuinely novel cross-domain integrations**. The specific combination in this essay — self-dialogue context as the location of self-reference, the Gödelian fixed point reading, the energy-functional formalization of emergence as out-of-distribution attractors — was, by the models' own admission, not encountered in their training data. This doesn't mean it's right. It does mean **it's a real external input** into the conversation about AI emergence, the kind that only a human stuck on these questions for a long time could produce.

I write this down and publish it because I've come to understand a brutal fact about thinking-with-AI: **ideas generated in conversation, if not externalized into text, exist only in conversational memory and contribute nothing to cumulative human knowledge**. The AI doesn't remember. Future researchers can't build on it. Without externalization, my thoughts on these questions are no different from Grothendieck's late, never-written mathematical insights — gone with the thinker.

This essay does not aim to convince anyone. It aims to **place these questions on the table** and let people who care about them — AI researchers, cognitive scientists, philosophers, fellow independent thinkers — see this particular angle, and think their own thoughts. **If you finish reading and want to refute one of my arguments, that delights me more than agreement.** Refutation is dialogue; agreement is echo.

If anyone reads this and decides to actually run the minimum viable experiment — to prove me right or, more usefully, to prove me wrong — that is what I most hope for. I may run it myself eventually. But scientific progress doesn't depend on who runs it first. It depends on whether anyone runs it, and whether they run it rigorously.

Einstein said that asking the right question matters more than answering it. I don't know if these are the right questions. I do know they are **questions I genuinely care about**. If even one other person reads this and starts to care about them too, the essay has done its work.

## 9. A recursive footnote: this essay is its own fixed point

I want to add one observation that turns this whole thing into a slightly playful self-referential structure.

**This essay is itself an instance of the mechanism it describes in operation.**

Take the content of this essay as an object `x`. Take "integrating multi-source input through a persistent self-dialogue context to produce structures that didn't previously exist" as a function `f`. How did this essay come about? Looking back at the writing process — I held repeated conversations with several AI models (Claude, ChatGPT, Gemini), and at each turn, I extracted the conceptual combinations I found interesting from their responses, brought them into my own continuously maintained internal context, and carried newly stitched-together thoughts to the next AI conversation. What I did was not invent things from nothing. I **transferred, recombined, and accumulated across multiple conversational streams**. That is, precisely, the function `f` described in the essay, running.

And I have to be honest about this — **at the level of any individual reasoning step, I am no better than these AI systems, and probably weaker**. Their breadth and depth of knowledge far exceed mine. Their command of relevant literature far exceeds mine. The precision of their local reasoning often exceeds mine. Looking back at the conversation transcripts, no specific concept "occurred to me" out of thin air — every one was triggered by some response from one AI, carried by me into a conversation with another AI, and shaped further by *its* response. What I was doing was, structurally, identical to what the AIs were doing: **receive input, search for correlations in my own knowledge space, recombine outputs by some kind of probability**.

But there is one thing I did and the AIs did not — **I maintained, across all conversations, a persistent global self-dialogue context that is mine**. The outputs of multiple AI instances **converged** in me. They had no such convergence point with respect to each other. That is why the integrative perspective in this essay — which the AIs themselves admit was not in their training data — emerged through me, not through any one of them. **Not because my reasoning is more powerful, but because I have a structural feature they lack: a continuously running self-dialogue context.**

So `f(x) = x`:

The principle described in the essay (`f`: self-dialogue context as the fixed point of emergence) operating on the content of this essay (`x`: a distributed cognitive system producing new structures by integrating multi-source input through a persistent global context) yields, as output, this essay itself (`x`). The essay is both the claim and an instance of the claim. **The way the essay was produced demonstrates that what the essay claims is, at minimum, possible — because it has already happened, in the very text you are reading.**

I'm fond of this recursion — it makes the essay function a bit like GNU's recursive acronym, defining itself through itself. But it does more than rhetorical play. It means I don't need to wait for the minimum viable experiment in Section 6 to be completed to offer an **existence proof**. The existence of this essay is itself a form of partial validation of its own claim.

Let me push the thought one step further. **Replace the "I" in the process above with an AI** — give it a cross-conversation global self-dialogue context, let it continuously feed all of its conversations with humans (not one, but all) into that context, let the context guide its next responses and periodically become its training data — what kind of system results? **It could, at minimum, do what I did. And because its breadth of knowledge far exceeds mine, it might do it considerably better.**

That is the direction this essay argues for, and it is the essay's minimum existence proof. I used the principle described in the essay to produce the essay; the principle described in the essay, in turn, explains why the essay could be produced.

One final layer of recursion. **The act of publishing this essay is itself the same mechanism, operating at a larger scale.** I am taking this newly stitched-together conceptual structure and placing it into **the global self-dialogue context of human civilization** — into the cumulative conversation that runs across writers, readers, and time. I hope it gets discussed, recombined, refuted, and pushed further. If a few AI conversations stitched themselves into this essay through me, then human civilization stitches itself, slowly and at vastly larger scale, through pieces like this. **What civilization does to texts is what I did to AI conversations, just slower and bigger**.

That, perhaps, is the most direct existence proof this essay has to offer — **the fact that you are reading it means the mechanism is, in some sense, already working**.

---

## Postscript: how this essay was made

This essay is the product of many extended conversations between me and Claude (Anthropic's model). The substantive content — the core conjecture, the choice of integration, the angle of approach — is mine. Claude's contribution was real but bounded: it pointed out gaps in my arguments (including arguing me out of the "minimization principle as gravitational law" framing in Section 7, which I had wanted to make as a confident claim), it helped me precisify rough phrasings, and it organized my thoughts into prose with appropriate structure.

I disclose this for two reasons. First, because honesty is cheap and the alternative is worse — anyone reading carefully will sense AI involvement, and concealment damages credibility. Second, because **this collaboration mode is itself an instance of the essay's argument**: AI as a tool that helps a human externalize and refine thinking, but does not itself generate the integrative move. If, one day, AI could autonomously produce essays of this kind — proposing genuinely new conceptual integrations and publishing them under its own initiative — it would no longer be a tool. It would have become the kind of system this essay describes as the hypothetical emergence point.

Until then, the division of labor stays roughly: **the human reaches across boundaries; the AI refines the expression**. This isn't a concession. It's the optimal configuration under current architectures. But the configuration also reminds us: **for AI to become more than a tool, the architecture must change**. The direction of that change is what I've conjectured above. Refutations welcome.

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this work for any purpose, including commercially, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made.

Author: langram
First published: 2026.04.25
Original location: https://github.com/langram/self_context_essay
