# Intent Classifier

Compare three intent-classification approaches on the same customer
messages: local sentence embeddings, TypeSafe JEV, and OpenAI frontier models.
Each result shows the predicted intent, confidence or score, and model details.

Tick the models to compare and click **Run selected**. The OpenAI list shows
the text models available to your API key. OpenAI and JEV requests are billed
per call.

## Results

Switch the results to **Table** for a side-by-side comparison: top two intents,
whether the answer is correct (for the built-in examples), input/output tokens,
and latency for each model.

- **Download CSV** saves one line per request, so averages and percentiles can
  be recomputed elsewhere.
- **Full size** opens the table on its own at full page width, with the run
  date and the messages above it. **Print** prints it in landscape (or save it
  as PDF from the print dialog). **Close full size** or Escape returns to the
  normal view. Printing from the normal view also hides the controls and
  prints just the results.

Embedding latency counts only encoding the message and matching it to the
intents; loading the model and embedding the intent descriptions is one-time
setup and is excluded. Remote latency is the full API round trip.

## JEV: all intents or a shortlist

JEV scores every intent option it is sent, so its tokens grow with the number
of options (about 300 fixed input tokens plus ~10 input and ~10 output per
option). JEV is offered two ways:

- **TypeSafe JEV · all 77 intents** sends every intent.
- **TypeSafe JEV · top 15 shortlist** lets the local embedding model pick the
  closest intents first, using about 55% fewer input and 80% fewer output
  tokens. If the right intent is not in the shortlist, JEV cannot choose it.

Set the shortlist size with `JEV_SHORTLIST` in `.env` (1 to 76; default 15). Shortlist latency
includes the local shortlist step (embedding the message to pick the intents),
so it can be higher than the all-77 option even though JEV itself receives
fewer tokens. The step takes longer for long messages and when other requests
are running at the same time. Its time is shown as a ⚠ warning under the
latency in the table and on the card, and in the CSV as
`shortlist_ms_included`. The one-time embedding model load is excluded.
The raw JEV response lists the options sent under `optionsSent`.

## Batches and average latency

Put several messages in the box, one per line (Shift+Enter adds a line;
**All examples, one per line** fills in every example).

- **Repeat each message** sends each message that many times. Repeating the
  same message can make OpenAI look faster than real traffic because of prompt
  caching, so prefer several different messages.
- **Parallel requests** sends that many requests at once per model to simulate
  concurrent load. Keep it at 1 for clean per-request timings. Up to 3 remote
  models also run at the same time.

With more than one request, the table shows average latency with p50 and p95,
average tokens per request, and how many known examples were
answered correctly. The page shows how many requests a run will send before
you start. Every OpenAI and JEV request is billed.

## Example result

A single run on 11 Oct 2026 (one request per model, not averaged) with this
message:

> My card got stuck in an overseas ATM while I was traveling, and the screen
> timed out before returning it. I don't think anyone stole my PIN, but I can't
> retrieve the physical card and I leave for my next flight in four hours. Do I
> need to report this as stolen to block it, or can you just issue a temporary
> digital replacement to my Apple Wallet so I can pay for my hotel tonight?

| Model | Top intent | 2nd intent | Input tokens | Output tokens | Latency |
|---|---|---|---:|---:|---:|
| Embedding · all-MiniLM-L6-v2 | lost_or_stolen_card (0.542) · *abstained* | compromised_card (0.520) | 88 | n/a | 85 ms (35,421 ms setup excluded) |
| JEV · jev-1.13.0 · all 77 | card_swallowed (0.940) | lost_or_stolen_card (0.050) | 1,101 | 828 | 319 ms |
| JEV · jev-1.13.0 · top 15 | card_swallowed (0.980) | lost_or_stolen_card (0.010) | 513 | 178 | 408 ms (incl. 115 ms shortlist step) |
| OpenAI · gpt-4.1 | card_swallowed (0.970) | lost_or_stolen_card (0.900) | 607 | 228 | 2,795 ms |
| OpenAI · gpt-6-luna | card_swallowed (0.990) | lost_or_stolen_card (0.940) | 605 | 348 (168 reasoning) | 3,761 ms |

"Correct?" is blank in the app for this run because the message is not one of
the built-in examples.

What it shows:

- JEV and both OpenAI models chose `card_swallowed`. The embedding model's top
  two scores were too close (0.542 vs 0.520), so it abstained.
- The JEV shortlist gave the same answer with 53% fewer input and 79% fewer
  output tokens. Its latency was higher only because of the 115 ms local
  shortlist step; without it, the JEV call took about 293 ms, against 319 ms
  for all 77.
- JEV separated the top two intents much more sharply (0.94 vs 0.05) than the
  OpenAI models (0.97 vs 0.90 and 0.99 vs 0.94).
- JEV (all 77) answered roughly 9 to 12 times faster than the OpenAI models.

Screenshots of this run and of a batch test (cards, full size and print views)
are in [results/](results/README.md).

## Run

Create the virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Then start the app either way:

```powershell
.\.venv\Scripts\python app.py
```

Or:

```powershell
.\.venv\Scripts\python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

Open http://localhost:8001. Stop with Ctrl+C.

External classifiers read credentials from `.env` (copy `.env.example`). Set
`JEV_KEY` for JEV and `OPENAI_API_KEY` for OpenAI; `JEV_MODEL` and
`JEV_SHORTLIST` are optional, and `FORCE_MOCK=1` makes JEV return a mock result
without calling the API. OpenAI model choices are discovered from the
models available to that key; the key itself is only used by the server and is
not sent to the browser.

## Files

| File | What it does |
|---|---|
| `classifier.py` | Local embedding classifier and available embedding models |
| `jev_classifier.py` | TypeSafe System One Choice API adapter |
| `llm_classifier.py` | OpenAI Responses API adapter and model discovery |
| `app.py` | Web server, model catalog, `/classify` endpoint, and token usage |
| `index.html` | Model selection, batch settings, results cards, table, CSV and print view |
| `labels.json` | The 77 intent names and classification choices |
| `requirements.txt` | Python packages needed |
| `.env.example` | Template for `.env` (API keys and JEV settings) |
| `results/` | Screenshots and notes from example runs |
| `.cache/` | Downloaded local embedding models; created automatically |
