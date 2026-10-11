# Does Every Decision in an AI Agent Need an LLM?

*Exploring the middle ground between traditional classifiers, generative AI, and specialized decision models.*

## Where We Started

For years, the answer to "What does this customer want?" was a classifier.

Define the categories, train the model, and route the request. Delivery vs. invoice vs. complaint. Efficient, predictable, and sometimes a little rigid.

Then came LLMs.

Suddenly, we had models that could understand context, reason through ambiguity, and handle messy, open-ended language.

And naturally, we started building agents around them.

But something interesting happens when we look inside an agent.

An agent doesn't just generate an answer. It navigates through multiple steps, making decisions along the way.

Which tool should I call? Is the request complete? Do I need additional information? Should I proceed or escalate?

Each decision might be small. But together, they determine how the entire workflow behaves.

**And that raises an interesting question: Does every decision in an AI agent really need an LLM?**

## Let's Take a Simple Example

Consider a hypothetical SAP sales order scenario.

Imagine a customer sends an email:

> Hi, I know my order is already being prepared for shipment, but we don't need all 100 units anymore. Can you do something about the remaining 40? I don't want to delay the rest of the delivery.

This isn't necessarily a straightforward sales order change request.

The customer wants fewer units but also wants to protect the existing delivery schedule. Depending on the order and shipment status, several actions might be possible.

An AI agent supporting this process would need to navigate multiple decisions.

| Step | Question | Possible action |
|---|---|---|
| 1. Understand intent | What is the customer asking? | Identify quantity reduction and delivery constraints |
| 2. Retrieve context | What is the current order and delivery status? | Fetch sales order and delivery details from SAP |
| 3. Evaluate options | What is the appropriate next step? | Request clarification, investigate a change, or escalate |
| 4. Validate | Is the proposed change permitted? | Apply SAP business rules and authorization checks |
| 5. Execute | What action is authorized? | Initiate an allowed change or route for human review |
| 6. Respond | What should we communicate? | Explain the outcome to the customer |

Here's where the distinction becomes interesting.

An LLM could interpret the email and extract the customer's requirements.

A specialized decision model like **TypeSafe Jev** could evaluate the message alongside the retrieved business context and recommend the most appropriate next step from a predefined set of options.

But Jev wouldn't determine whether SAP permits a quantity change after delivery processing has started. That decision belongs to the underlying business rules and transaction state.

The agent orchestrator would coordinate the process, and an LLM could generate the final customer-facing response.

**Some steps require language understanding. Others require structured judgment. And some should remain entirely deterministic.**

![The sales order example as a flow. An email asking to cut an order by 40 units without delaying delivery passes through six steps inside the agent orchestrator: an LLM understands the intent, the orchestrator retrieves SAP context, a decision model such as TypeSafe Jev chooses the next step from predefined options, SAP business rules validate the change, SAP or a human executes it, and an LLM writes the reply.](images/processflow.png)

So why should we automatically reach for a generative LLM at every decision point?

## The Quiet Cost: Tokenomics

As agents become more complex, so does the number of decisions they make.

And when those decisions repeatedly involve LLM calls, the cost isn't just tokens.

There's latency, compute, and the operational complexity of managing those calls.

Of course, a well-designed agent can make multiple decisions within a single LLM invocation. But that doesn't eliminate the underlying architectural question.

**Just because an LLM can make a decision, does that mean it's the right tool for it?**

## The Middle Ground

Technology has a funny way of moving like a pendulum.

We start with something structured and predictable. Then a powerful new technology arrives, and suddenly we want to use it everywhere.

Eventually, we begin rediscovering the value of simpler, more specialized approaches.

I wonder if we're seeing something similar in AI architecture.

That's why specialized decision approaches, including **TypeSafe Jev** and **Strands Decider**, caught my attention.

Rather than focusing on generating lengthy responses, these approaches are designed around evaluating context and making constrained decisions.

For example, TypeSafe Jev offers different decision patterns:

- **Choice:** Which option best fits the available context?
- **Score:** How strongly does something match a defined scale?
- **Noul:** Does a particular condition appear to be satisfied?

Think back to our sales order scenario.

Instead of asking a generative LLM to decide every next action, we could evaluate a predefined set of alternatives using a specialized decision model.

The orchestrator would still coordinate the workflow. SAP would remain responsible for business validations and transaction execution. And an LLM could handle tasks where flexible language understanding or generation adds value.

This doesn't mean replacing LLMs with Jev everywhere.

It means recognizing that **decision-making and text generation are not necessarily the same problem**.

And perhaps the pendulum isn't swinging backward toward traditional classifiers. Perhaps it's finding a more practical middle ground where different types of intelligence work together.

![So what should we use? A simple architecture lens: use a classifier for known categories and stable tasks, a decision model for structured decisions and scores, and an LLM for reasoning, extraction and open-ended output. The goal is to choose the right intelligence for the job.](images/choosing-the-right-ai-architecture.png)

## But Does It Hold Up?

An interesting architectural idea is one thing. Demonstrating that it works is another.

For an enterprise solution, I would evaluate four dimensions:

**Accuracy:** Does the approach consistently make the correct decision, especially when requests are ambiguous or similar?

**Latency:** How quickly can it produce a usable decision?

**Cost:** What does each decision cost, and how does that scale with request volume?

**Privacy:** Where is the data processed, and what information needs to leave the enterprise environment?

A specialized decision model isn't automatically the better option.

If a traditional classifier, local embeddings, or a few deterministic rules can reliably solve the problem, why introduce another model?

And if a decision requires deeper reasoning, complex context, or open-ended generation, an LLM might still be the better choice.

The answer should come from evidence, not assumptions.

## My Learnings: Putting the Idea into Practice

In any enterprise, so much starts with the voice of a customer or supplier: an email, a chat, a ticket, someone who needs an answer.

And the challenge isn't always understanding what they're saying. It's deciding what to do next.

I chose the public **BANKING77 dataset** because its 77 banking intents offer an interesting classification challenge. Customer requests can sound very similar while representing different intents.

So I built a small application that takes the same customer message and compares three approaches side by side:

1. **Local embeddings** — using semantic similarity to identify the most likely intent.
2. **TypeSafe Jev** — using a specialized decision model to select from predefined intent categories.
3. **OpenAI models** — using an LLM-based approach to classify the request.

The objective is to compare **classification accuracy, latency, and cost** across these approaches.

| Approach | Accuracy | Average latency | Cost per 1,000 requests |
|---|---|---|---|
| Local embeddings | To be measured | To be measured | To be measured |
| TypeSafe Jev | To be measured | To be measured | To be measured |
| OpenAI models | To be measured | To be measured | To be measured |

This is a focused intent-classification experiment, not a complete multi-step agent.

But intent classification is one of the decisions an agent may need to make. It provides a useful starting point for exploring whether every decision requires a generative model.

I'm particularly interested in the trade-offs.

Does a specialized decision model deliver comparable accuracy with lower latency or cost? Can local embeddings achieve similar results without an external model call? And when does the additional flexibility of an LLM justify its overhead?

The measurements will help answer those questions.

I'll share the implementation, evaluation methodology, and results on GitHub.

**GitHub:** [https://github.com/PrasanthiDesiraju/decisionModelJEV.git]

## Where I Landed

LLMs are probably the closest thing to a magic wand we've had in a while.

But even magic comes with a token bill! :)

The answer isn't to replace every classifier with an LLM, or every LLM with a specialized decision model.

It's about learning, unlearning, and relearning as technology evolves.

I think of architecture a little like Jenga. Every piece has its place, and every choice affects the stability of the overall structure.

Sometimes the smartest decision is knowing which piece you don't need.

**The goal was never to use the smartest model. It's to choose the right intelligence for the job.**

And perhaps that's an important part of building the next generation of enterprise AI agents.

---

#AgenticAI #EnterpriseArchitecture #LLM #AIEngineering #DecisionModels #SAPBTP
