# Step into World Models

The code behind the [Understanding World Models](https://stepinto.dev/series/understanding-world-models/) series on stepinto.dev, and the talk of the same name. One simulated room with a heater, a door nobody logs, and plain TypeScript with no ML libraries. Every model trains in under a second.

There's a [playground](https://stepinto.dev/heater/) if you'd rather run it all in a browser.

## What's here

Each script prints every number in its post. Add `csv` to also write the post's data to `static/data/`, then run the matching Python script to draw the charts into `static/charts/`.

| Script | Post | What it does |
|---|---|---|
| `tools/heater_fit.ts` | [103, Learning the Heater From Its Own Logs](https://stepinto.dev/posts/world-models-103/) | Records 4,800 logged steps from a simulated room and fits four weights by gradient descent |
| `tools/heater_door_fit.ts` | [104, Latent State Is the Fields You Never Logged](https://stepinto.dev/posts/world-models-104/) | Adds a door the logger never records, and a model with one private field trained by backpropagation through time |
| `tools/heater_llm.ts` | [106, How a World Model Differs From an LLM](https://stepinto.dev/posts/world-models-106/) | Asks Claude the same questions, with and without the logs and with the 104 model as a tool, then scores the answers |
| `tools/heater_plan.ts` | [107, Planning Against the Model](https://stepinto.dev/posts/world-models-107/) | The physics-shaped model, then exhaustive search, random shooting, the cross-entropy method and replanning every step |
| `tools/heater_band.ts` | [108, When the Model Doesn't Know](https://stepinto.dev/posts/world-models-108/) | Eight bootstrap copies of the model, the band they draw, and a planner told the coldest copy |
| `tools/heater_monitor.ts` | [109, The Model as a Monitor](https://stepinto.dev/posts/world-models-109/) | Runs the model beside the room as a fault monitor for a dead heater, a stuck one and an open window |

`tools/heater_charts.py` draws the 103 and 104 charts and holds the shared chart style the other `*_charts.py` scripts import.

## Running it

You need Node 23 or newer, which runs TypeScript directly, and Python 3 for the charts. Run everything from the repo root.

```sh
node --no-warnings tools/heater_fit.ts
node --no-warnings tools/heater_plan.ts csv
python3 tools/heater_plan_charts.py
```

`heater_llm.ts` is the only script that calls an API. It needs the Anthropic SDK and an `ANTHROPIC_API_KEY`, and the instructions are at the top of the file. A full run cost about three and a half dollars in October 2026. The answers from the run in the post are already in `static/data/heater-llm-runs.csv`, so you can draw its charts without making any calls.

The scripts share a seeded random number generator, so the weights and numbers should match the posts.

## Licence

MIT. See [LICENSE](LICENSE).
