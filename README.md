# Topic Wordlist Generator

Complete usage guide for `topic_wordlist.py` **v1.5.0**. Updated October 22, 2026.

Generate short topical words and names with a local Ollama model. Examples below cover broad sports, worldwide cars, and NFL teams and players, including Pro Football Hall of Fame players. The same script works with other topics; these examples do not require sports-specific code.

**Recommended starting model for this guide:** `qwen3.5:9b-q4_K_M`. This is a practical candidate under the laptop size constraint, not a model proven to produce exhaustive inventories.

## Contents

1. [What the script produces](#1-what-the-script-produces)
2. [Install and check your setup](#2-install-and-check-your-setup)
3. [Choose a run size](#3-choose-a-run-size)
4. [Broad sports wordlist](#4-broad-sports-wordlist)
5. [Worldwide cars wordlist](#5-worldwide-cars-wordlist)
6. [NFL teams, players, and Hall of Famers](#6-nfl-teams-players-and-hall-of-famers)
7. [Pause, resume, and expand](#7-pause-resume-and-expand)
8. [Compare or switch models](#8-compare-or-switch-models)
9. [Control length and formatting](#9-control-length-and-formatting)
10. [Use categories, seeds, and exclusions](#10-use-categories-seeds-and-exclusions)
11. [Read progress and diagnose low yield](#11-read-progress-and-diagnose-low-yield)
12. [Troubleshooting](#12-troubleshooting)
13. [Complete option reference](#13-complete-option-reference)
14. [Optional OpenAI backend](#14-optional-openai-backend)
15. [Other topics and combining files](#15-other-topics-and-combining-files)
16. [References and validation](#16-references-and-validation)

## 1. What the script produces

The output is an alphabetized UTF-8 `.txt` file, one entry per line. By default, it creates lowercase versions with spaces and compact versions without punctuation or spaces. For example:

```text
baltimore ravens
baltimoreravens
ford mustang
fordmustang
```

Those four lines represent two original terms. Additional spacing and capitalization variants increase the number of output lines without adding new factual coverage.

Each run also creates a checkpoint next to the output:

| File | Purpose |
| --- | --- |
| `sports_qwen.txt` | The usable wordlist. |
| `sports_qwen.txt.state.json` | Saved settings, planning, raw accepted terms, completed rounds, partial batches, and diagnostics. Keep this to resume. |
| `sports_qwen.txt.lock` | Temporary protection against two processes writing the same output. Normally removed when the script exits. |

### Coverage expectations

The local Ollama path generates from the model's learned knowledge. It does **not** browse websites, download rosters, or query a vehicle catalog. Asking for "current players" expresses the desired scope; it does not verify the current roster.

More categories and rounds provide additional opportunities for recall. They do not guarantee all players, all vehicle models, or all Hall of Famers. Names may be missing, incorrectly associated, or invented. For an inventory that must be complete, collect authoritative source data and use it as reviewed seed material.

This is a flat wordlist, not a team-to-player or make-to-model database. The checkpoint records which generation category produced a term, but that is not verified relationship data.

## 2. Install and check your setup

### Requirements

- Python **3.10 or newer**.
- `topic_wordlist.py` v1.5.0 for category expansion and the thorough-mode examples.
- Ollama installed and running, with the chosen model downloaded.
- No third-party Python packages: there is no `pip install` step for this script.

### Windows setup

Save the script in `C:\PenTesting`. In **Command Prompt**:

```bat
cd /d C:\PenTesting
python3 --version
python3 topic_wordlist.py --version
ollama --version
ollama pull qwen3.5:9b-q4_K_M
ollama list
```

The script version should report `1.5.0`. If your Windows Python command is `python` or `py -3` instead of `python3`, use that prefix consistently.

The [official Ollama model listing](https://ollama.com/library/qwen3.5:9b-q4_K_M) lists this Q4_K_M package at approximately **6.6 GB**. Download size is not total runtime memory usage.

Open the Ollama application. If you run Ollama manually and its server is not already running, start this in another terminal:

```bat
ollama serve
```

If that says the port is already in use, the existing Ollama server may already be running; do not start a second server on the same port.

### Shell notes

All generator commands below are single-line commands. They work in Command Prompt, PowerShell, or a Unix-like shell when Python and Ollama are available in that environment.

- In PowerShell, change directories with `Set-Location C:\PenTesting` rather than `cd /d`.
- In WSL, use `cd /mnt/c/PenTesting` if that is where your script lives.
- Running both Python and Ollama on Windows is the simplest match for the examples. WSL-to-Windows networking may require a reachable `--ollama-url` instead of the default localhost address.
- Use ordinary straight quotes around topics, scopes, and paths containing spaces.

### Verify GPU loading

While generation is running, open another terminal:

```bat
ollama ps
```

According to the [Ollama FAQ](https://docs.ollama.com/faq), `100% GPU` means the model is fully GPU-loaded. A CPU/GPU split means it uses both. This is a resource check, not a measure of output quality.

## 3. Choose a run size

These are suggested starting profiles, not output guarantees. All retain an 8,192-token context, a 2,048-token response limit, and batches of 20 terms.

| Profile | Categories | Rounds | Terms per category per round | Use |
| --- | ---: | ---: | ---: | --- |
| Trial | 14 | 1 | 25 | Check relevance and runtime before committing to a broad run. |
| Broad | 96 | 4 | 100 | The main automatic examples in this guide. |
| Expansion | 128 | 6 | 100 | Add breadth and further passes after reviewing a broad run. |

`--categories` is a planning budget, not the number of terms and not a promise that the model will supply that many useful groups. With a list-style scope, the budget is shared across its coverage areas.

`--rounds` is the **total** number of passes per category. Increasing from 4 to 6 adds two passes for existing categories. Newly added categories receive all six passes.

`--per-category` is the target of new base terms per round. `--batch-size` is the maximum number requested in an individual Ollama call. For example, a target of 100 with batches of 20 can require multiple calls.

In thorough mode, a productive short reply does not end the round. Generation stops when it reaches the target, receives two consecutive replies with no new terms, or reaches its batch budget. Output-validation failures have their own bounded retries and can still stop the run.

For Ollama, the successful-batch cap per round is `max(8, 3 * ceil(target / initial_batch_size))`. The 100/20 profile permits up to 15 successful batches per round, plus failed-output retries. Planning is also bounded. A broad run can therefore take substantially longer than the first trial.

## 4. Broad sports wordlist

This asks for sports, leagues, teams, nicknames, mascots, players, and stadiums as separate coverage areas. Commas matter: `players, stadiums` creates separate areas; `players and stadiums` remains one area.

### Trial

```bat
python3 topic_wordlist.py "sports" --scope "Sports, leagues, teams, nicknames, mascots, players, stadiums" --model qwen3.5:9b-q4_K_M --think off --categories 14 --rounds 1 --per-category 25 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --thorough -o sports_trial.txt
```

Inspect the result for actual names and useful vocabulary. If the model mostly returns vague labels or repeated familiar names, extending the same run may have limited value.

### Broad run

```bat
python3 topic_wordlist.py "sports" --scope "Sports, leagues, teams, nicknames, mascots, players, stadiums" --model qwen3.5:9b-q4_K_M --think off --categories 96 --rounds 4 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --thorough -o sports_qwen.txt
```

Each filename starts a separate run. To expand the trial instead of starting the broad run from scratch:

```bat
python3 topic_wordlist.py -o sports_trial.txt --resume --categories 96 --rounds 4 --per-category 100 --thorough
```

Existing completed trial rounds remain completed. Raising the target does not refill them; later rounds use the new target.

## 5. Worldwide cars wordlist

This requests manufacturers, makes, current and discontinued models, trims, classic cars, and discontinued brands across regions. The scope supplies the requested types of names; it does not activate a built-in vehicle database.

```bat
python3 topic_wordlist.py "cars worldwide" --scope "Manufacturers, makes, current production models, discontinued models, trims, classic cars, discontinued brands" --model qwen3.5:9b-q4_K_M --think off --categories 96 --rounds 4 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --entities-only --thorough -o cars_qwen.txt
```

`--entities-only` asks for names and enables the stricter inventory-label filter. It can reject labels such as `model names`; it does not validate whether a model or trim exists.

If you know which manufacturers you need, you can bypass automatic planning for a focused run:

```bat
python3 topic_wordlist.py "cars" --scope "Actual model names and trim names for the specified manufacturers; include current and discontinued vehicles" --category Ford --category Toyota --category BMW --category Chevrolet --model qwen3.5:9b-q4_K_M --think off --rounds 4 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --entities-only --thorough -o cars_selected_makes.txt
```

Those four manufacturers are examples, not a worldwide manufacturer inventory. The same technique works with your own `--categories-file` containing manufacturers, product families, places, or other parent groups.

## 6. NFL teams, players, and Hall of Famers

Here, "Hall of Famers" means **players inducted into the Pro Football Hall of Fame in Canton**. It does not mean college Hall of Fame members, team honor rolls, nominees, or people predicted to be inducted later.

Use the [official Hall of Fame directory](https://www.profootballhof.com/players) to verify membership. Its directory also includes non-player contributors; check the individual's role when building a players-only seed file. Current affiliations should be checked against official team rosters, accessible through the [NFL team directory](https://www.nfl.com/teams/).

### A. Automatic NFL coverage

```bat
python3 topic_wordlist.py "NFL teams and players" --scope "NFL team names, historic NFL franchise names, current NFL players, retired NFL players, inducted Pro Football Hall of Fame players, established team nicknames, established player nicknames" --model qwen3.5:9b-q4_K_M --think off --categories 96 --rounds 4 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --entities-only --thorough -o nfl_players_qwen.txt
```

This asks the model to plan the groups itself. It does not guarantee that every franchise, position, or era receives equal coverage.

### B. Explicit team-by-team coverage

Use this alternative when you want every listed franchise to receive its own generation requests. It uses the script's existing generic category-file feature.

Create `nfl_teams.txt` in the same directory as the script. Save the following names as plain UTF-8 text, one per line, with no Markdown fences. These 32 franchise names were checked against the team table in the [2026 NFL rulebook](https://operations.nfl.com/rules-officiating/2026-nfl-rulebook) on October 22, 2026.

```text
Arizona Cardinals
Atlanta Falcons
Baltimore Ravens
Buffalo Bills
Carolina Panthers
Chicago Bears
Cincinnati Bengals
Cleveland Browns
Dallas Cowboys
Denver Broncos
Detroit Lions
Green Bay Packers
Houston Texans
Indianapolis Colts
Jacksonville Jaguars
Kansas City Chiefs
Las Vegas Raiders
Los Angeles Chargers
Los Angeles Rams
Miami Dolphins
Minnesota Vikings
New England Patriots
New Orleans Saints
New York Giants
New York Jets
Philadelphia Eagles
Pittsburgh Steelers
San Francisco 49ers
Seattle Seahawks
Tampa Bay Buccaneers
Tennessee Titans
Washington Commanders
```

Then run:

```bat
python3 topic_wordlist.py "NFL teams and players" --scope "For each franchise include its name, established nicknames, and current and retired players across positions and eras. Include players from its former city or team names. Include inducted Pro Football Hall of Fame players; exclude coaches, owners, and prospective inductees from the Hall of Fame category." --categories-file nfl_teams.txt --seed-file nfl_teams.txt --category "Historic NFL franchise names" --category "Pro Football Hall of Fame players" --model qwen3.5:9b-q4_K_M --think off --rounds 4 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --entities-only --thorough -o nfl_by_team_qwen.txt
```

This creates 34 explicit categories: 32 teams, historic franchise names, and Hall of Fame players. It also merges the 32 team names into the output as seeds, subject to the normal filters. The seed names do not depend on the model remembering to return them.

There is deliberately no `--categories` option in this command. Explicit category files bypass automatic category discovery; a numeric category budget does not add more categories to that plan.

This approach gives each team a turn. It does not guarantee a complete player history or verify that every generated player belonged to the requested franchise. Players associated with several teams are deduplicated in the final flat list.

For a team-only result, make a separate run with that team as `--category` and its own output filename. The combined file does not retain team labels next to each player.

### C. Hall of Fame players only

Use a separate run if you want to review Hall of Fame results without mixing them with other NFL players:

```bat
python3 topic_wordlist.py "Pro Football Hall of Fame players" --scope "Players actually inducted into the Pro Football Hall of Fame in Canton. Include established short name variants. Exclude college-only honorees and nominees and prospective inductees. Exclude coaches and owners and other non-player contributors." --model qwen3.5:9b-q4_K_M --think off --categories 24 --rounds 3 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --entities-only --thorough -o nfl_hall_of_famers_qwen.txt
```

Do not interpret this output as a certified membership list. The model can have an outdated view of recent induction classes. A reviewed seed file from the official directory is the way to ensure known members are present.

## 7. Pause, resume, and expand

### Pause and resume

Press `Ctrl+C`. Validated batches and completed rounds are checkpointed. Then use the **same output filename**:

```bat
python3 topic_wordlist.py -o sports_qwen.txt --resume
python3 topic_wordlist.py -o cars_qwen.txt --resume
python3 topic_wordlist.py -o nfl_players_qwen.txt --resume
```

Run only the command for the file you want to resume. Topic, scope, model, and most formatting/generation settings are recovered from its checkpoint.

### Expand an automatic plan

After reviewing the 96-category, four-round sports run:

```bat
python3 topic_wordlist.py -o sports_qwen.txt --resume --categories 128 --rounds 6 --thorough
```

With v1.5.0, this appends new categories to supported automatic v1.4+ plans while retaining existing category indexes, terms, and completed rounds. The model may still return fewer useful groups than requested.

Category counts and round counts cannot be reduced on resume. If your checkpoint already requests more than the example, omit that option or use a larger value.

### Extend the explicit NFL team plan

```bat
python3 topic_wordlist.py -o nfl_by_team_qwen.txt --resume --rounds 6 --thorough
```

Manual category plans can receive more rounds, but their category lists cannot be expanded with `--categories`. To change a manual list, create a new run and output filename using the revised category file.

### Change the term target

```bat
python3 topic_wordlist.py -o cars_qwen.txt --resume --rounds 6 --per-category 150 --thorough
```

This changes the target for newly started rounds. An unfinished round retains the target stored when that round began; completed rounds are not rerun or refilled.

### Preview a plan

For a new automatic run, append `--plan-only` to its full starting command. Planning is saved and printed without term-generation rounds. Start generation later with the same output path and `--resume`, omitting `--plan-only`.

Previewing is useful for seeing whether your intended types of names received sensible groups. It does not verify the plan's factual accuracy. Editing the scope or supplying a different manual plan requires a new output filename.

## 8. Compare or switch models

For a useful model comparison, use the same script version, scope, categories, rounds, limits, and formatting in separate fresh output files. Change only `--model`, a model-compatible `--think` setting if needed, and the output filename. This lets each model build a fresh plan and keeps its results separate.

You can also switch models while preserving progress:

```bat
python3 topic_wordlist.py -o sports2.txt --resume --model qwen3.5:9b-q4_K_M --think off --rounds 6 --batch-size 20 --context 8192 --max-tokens 2048 --thorough
```

Use a round total above the already completed/requested total if you want additional passes. A model switch alone does **not** rerun completed rounds or replace the saved plan. This produces a mixed-model result, not a clean comparison.

Model notes:

- Use `--think off` explicitly for the Qwen3.5 examples. The script's automatic family handling does not automatically disable thinking for every Qwen version.
- For your installed `gpt-oss:20b`, use `--think low`, not `--think off`. The script rejects boolean off/on for this family.
- Always pass `--model` in a new run when selecting the laptop model. Without an environment override, the script's legacy Ollama default is `qwen2.5-coder:32b`.
- Increasing context, token limits, or categories cannot establish that the model knows an omitted fact. Evaluate the names themselves, not just file size.

## 9. Control length and formatting

The main commands use `--max-words 3 --max-length 24`, matching the preference for short entries. Both limits apply to the original term **before** compact versions are produced. A long phrase is not accepted merely because removing its spaces would make it fit.

Overlong names are omitted, not mechanically chopped into arbitrary fragments. The generation prompt asks for established short aliases when available. Strict limits can also exclude legitimate names with several parts.

### Export readable names only

```bat
python3 topic_wordlist.py -o sports_qwen.txt --export-only --styles words --case lower
```

### Export compact names only

```bat
python3 topic_wordlist.py -o sports_qwen.txt --export-only --styles compact --case lower
```

### Export additional formatting variants

```bat
python3 topic_wordlist.py -o sports_qwen.txt --export-only --styles words,compact,hyphen,underscore --case all
```

### Allow longer names

```bat
python3 topic_wordlist.py -o nfl_players_qwen.txt --export-only --max-words 4 --max-length 40
```

`--export-only` makes no model calls. It rewrites the same `.txt` file and saves the new formatting settings in the checkpoint. Save a copy of the previous text file first if you want both versions.

Increasing limits can restore long terms that exist in the saved raw data. It cannot recover terms that the model never returned or that were rejected before being saved. For new generation with the higher limits, add rounds:

```bat
python3 topic_wordlist.py -o nfl_players_qwen.txt --resume --rounds 6 --max-words 4 --max-length 40
```

Formatting can multiply output lines considerably. Review distinct base-term counts when judging knowledge coverage.

## 10. Use categories, seeds, and exclusions

All three accept UTF-8 text files with one entry per line. Blank lines and lines starting with `#` are ignored. These file flags are repeatable.

| Input | What it does | What it does not do |
| --- | --- | --- |
| `--categories-file groups.txt` | Supplies explicit generation categories and bypasses automatic planning. | Does not add those category headings to the final list automatically. |
| `--seed-file known_names.txt` | Merges known terms into the output, subject to formatting, length, and exclusion rules. | Does not turn the seeds into research notes or verify other model-generated terms. |
| `--exclude-file unwanted.txt` | Removes exact case-insensitive matches from exported terms/variants. | Does not remove every related spelling or word containing the excluded string. |

Seeds are not a retrieval database and are not automatically supplied as factual grounding to the generation prompt. Category files guide requests; seed files supply known output values. The NFL team example intentionally uses the same file for both roles.

A seed/exclusion illustration:

```text
# known_names.txt
Ford Mustang
Toyota Corolla
```

```text
# unwanted.txt
team names
teamnames
model list
modellist
```

Create those files before referencing them. A new run with reviewed car seeds could be:

```bat
python3 topic_wordlist.py "cars" --scope "Worldwide makes, models, trims, discontinued brands" --seed-file known_names.txt --exclude-file unwanted.txt --model qwen3.5:9b-q4_K_M --think off --categories 32 --rounds 3 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --entities-only --thorough -o cars_seeded.txt
```

The checkpoint stores loaded seeds and categories, so resume does not require their original files. Do not pass new `--seed-file`, `--category`, or `--categories-file` arguments with `--resume`; the script requires a new run for those changes.

Exclusions can be updated during an export:

```bat
python3 topic_wordlist.py -o sports_qwen.txt --export-only --exclude-file unwanted.txt
```

## 11. Read progress and diagnose low yield

The completion report shows unique base terms per group, rejected filler, duplicates, and a coverage label. `low yield` is a quantity signal; `unverified model inventory` is not a factual endorsement.

The checkpoint includes useful fields:

| Field | Meaning |
| --- | --- |
| `categories` | Actual saved generation categories, which may be fewer than the requested budget. |
| `coverage_areas` / `branch_plans` | Automatic planning stages and per-area groups. |
| `planning_diagnostics` | Bounded rejection details from planning attempts. |
| `jobs` | Completed category-round results. |
| `partials` | Accepted batches belonging to unfinished rounds. |
| `coverage_report` | Per-category counts from the most recent successfully completed run; it may be absent or stale after interruption. |
| `model_changes` | Model switches made while resuming. |

Inspect saved category and completed-round counts without changing anything:

```bat
python3 -c "import json; from pathlib import Path; s=json.loads(Path('sports_qwen.txt.state.json').read_text(encoding='utf-8')); print('Saved categories:',len(s.get('categories',[]))); print('Completed category-rounds:',len(s.get('jobs',{}))); print('Unfinished rounds:',len(s.get('partials',{})))"
```

If output is thin, check in this order:

1. **Plan:** Did the saved categories actually cover the requested areas? A numeric budget alone is not coverage.
2. **Yield:** Are later replies producing new relevant names or repeating earlier ones?
3. **Filters:** Are legitimate names too long for the selected word/character limits?
4. **Model:** Does a different model produce broader, more accurate results with comparable settings?
5. **Sources:** Does the task require an official roster, catalog, or register rather than recalled examples?

A `direct fallback` means group planning failed or produced no usable groups, so the script requested entries for the broader area directly. It keeps the run moving but may yield less detailed coverage.

## 12. Troubleshooting

| Symptom | Action |
| --- | --- |
| `unrecognized arguments: --thorough` | Check `python3 topic_wordlist.py --version`; use v1.5.0 for this guide. |
| Output/checkpoint already exists | Use the same output with `--resume`, or choose a new filename for a separate run. |
| Checkpoint not found | Confirm the `.txt.state.json` file exists beside the exact `.txt` path supplied to `-o`. A text list alone is not resumable. |
| Could not reach Ollama | Open the Ollama app or start its server; check the server address and downloaded model tag. |
| Model not found or unsupported model architecture | Check `ollama list`, pull the exact tag, and update/restart Ollama if the installed version cannot load it. |
| Truncated or schema-invalid model output | Keep thinking off for this Qwen profile, reduce batch size, and/or raise the response token limit before resuming. |
| Repeated duplicate batches | The model may have exhausted its accessible recall for that group. More rounds are not guaranteed to help. |
| No new output after a model switch | Completed rounds remain completed. Increase the total round count, expand a supported automatic plan, or start a fresh comparison run. |
| Cannot change scope/topic/backend on resume | These define the run. Use a new output filename. |
| Cannot expand a manual or older plan | Increase rounds, or create a new run with the desired category list. Numeric category expansion supports automatic v1.4+ plans. |
| Lock exists | Check whether another process is still using that output. Only remove a stale `.lock` after confirming no writer is running. |

A smaller-batch recovery command for an existing Qwen run:

```bat
python3 topic_wordlist.py -o sports_qwen.txt --resume --think off --batch-size 10 --context 8192 --max-tokens 4096
```

The script requires an Ollama context of at least 4,096 and at least 2,048 tokens left outside the response budget. With `--context 8192`, `--max-tokens 4096` satisfies that check. It is not a guarantee that every prompt will fit or that a model will follow the output format.

An unfinished round may retain a smaller saved batch size after previous backoff. Increasing `--batch-size` does not necessarily enlarge that partial round's next request.

To rebuild the `.txt` file from accepted checkpoint data without running a model:

```bat
python3 topic_wordlist.py -o sports_qwen.txt --export-only
```

That operation cannot reconstruct work that never reached the checkpoint. Keep the output and checkpoint together when moving or backing up a run.

## 13. Complete option reference

Run `python3 topic_wordlist.py --help` for the installed version's help text. Defaults below describe a new v1.5.0 run; saved settings generally take precedence on resume unless explicitly overridden.

### Topic and planning

| Option | Default / behavior |
| --- | --- |
| `topic` | Required for a new run; quote multiword topics. Optional with resume/export. |
| `-o`, `--output` | Defaults to a topic-based filename. Must end in `.txt`. Explicit filenames are recommended. |
| `--scope` | Broad current/historical associations. Short comma/semicolon lists of 2–16 items can become separate coverage areas; otherwise the model identifies areas. |
| `--language` | `English`. |
| `--categories` | `12`; maximum `500`. Automatic planning budget shared across areas. May increase on resume for supported automatic plans. |
| `--category` | Repeatable explicit category; bypasses automatic planning. |
| `--categories-file` | Repeatable UTF-8 category file; bypasses automatic planning. |
| `--plan-only` | Save/print a plan without term-generation rounds. |
| `--entities-only`, `--no-entities-only` | Default off; otherwise requests named entities and applies stricter generic-label filtering. |

An explicit list-style scope needs a category budget at least as large as its number of coverage areas. The model's inferred areas and groups are still subject to validation and planning limits.

### Generation

| Option | Default / behavior |
| --- | --- |
| `--rounds` | `2`; maximum `100`. Total passes per category. May increase, not decrease, on resume. |
| `--per-category` | `100`; maximum `1000`. Target new base terms per category-round. |
| `--batch-size` | `25`; range `1–100`. Ollama terms per request; main examples explicitly use `20`. |
| `--thorough`, `--no-thorough` | Default on. Continue productive short replies. Off restores ending an Ollama round at its first short reply. |
| `--output-retries` | `3`; range `0–8`. Retries malformed/truncated Ollama output with smaller batches; also controls planning retries. |
| `--plurals`, `--no-plurals` | Default on. Request natural singular/plural common nouns. Does not mechanically add suffixes to every term. |
| `--seed-file` | Repeatable files of terms to merge into output. |
| `--exclude-file` | Repeatable files of exact case-insensitive exclusions. |

### Model and connection

| Option | Default / behavior |
| --- | --- |
| `--backend` | `ollama`; alternative `openai`. Cannot switch backend on resume. |
| `--model` | Ollama: `OLLAMA_MODEL`, otherwise legacy `qwen2.5-coder:32b`. OpenAI: `OPENAI_MODEL`, otherwise the script's configured default. The guide always specifies its intended local model. |
| `--ollama-url` | `OLLAMA_HOST`, otherwise `http://localhost:11434`. |
| `--think` | `auto`; also `off`, `on`, `low`, `medium`, `high`. Supported settings depend on the model. This flag controls Ollama only. |
| `--context` | `8192`; Ollama only. Minimum `4096`. |
| `--max-tokens` | New Ollama run: `4096`; OpenAI: `12000`. Main local examples use `2048`. |
| `--timeout` | `600` seconds; must be positive. |
| `--retries` | `2`; temporary HTTP/network retries. Separate from output-validation retries. |
| `--delay` | `0.2` seconds between HTTP requests; cannot be negative. |
| `--web`, `--no-web` | Default off. OpenAI-only research per category; unavailable with the Ollama backend. |

### Output and lifecycle

| Option | Default / behavior |
| --- | --- |
| `--max-words` | `3`; range `1–16`. Counted on the original term before formatting. |
| `--max-length` | `32` for new runs; main examples use `24`. Applies to original terms and output variants. |
| `--min-length` | `1`; applies to exported variants. |
| `--styles` | `words,compact`; available values: `words`, `compact`, `hyphen`, `underscore`. |
| `--case` | `lower`; alternatives: `upper`, `title`, `original`, `all`. |
| `--no-ascii-fold` | Disable additional accent-folded Latin spellings. |
| `--resume` | Continue the checkpoint beside the selected output. |
| `--export-only` | Rewrite output from saved data without API calls. |
| `--version` | Show script version. |
| `-h`, `--help` | Show command help. |

General normalization also rejects malformed/control-character entries and limits a raw term to 200 characters and 16 whitespace-separated words. Raising `--max-length` above that does not remove the normalization bound.

Topic, scope, backend, language, web mode, plural settings, and supplied seeds/manual categories are fixed for a saved run. Start a new output file when changing those. Runtime timeout, HTTP retry count, and delay are command defaults rather than remembered run settings; repeat custom values when resuming if needed.

## 14. Optional OpenAI backend

The script also implements an OpenAI backend. It requires an `OPENAI_API_KEY` environment variable and uses the Responses API. It is not used by any local-model example above.

Its `--web` option performs research once per category and saves research notes and source URLs for subsequent generation. It does not verify every emitted term. To use it, start a new run with `--backend openai`, an API model available to your account, and optionally `--web`; omit Ollama-specific `--think`. The backend cannot be changed inside an existing Ollama checkpoint.

Ollama's batch continuation is not a promise of identical behavior on the OpenAI backend: v1.5.0 uses one term-generation request per category-round there. The local examples and batch limits in this guide specifically target Ollama.

Consult the [OpenAI Responses API documentation](https://developers.openai.com/api/docs/guides/migrate-to-responses) and [web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) for account/model requirements and current API behavior.

## 15. Other topics and combining files

To use another topic, change the topic, scope, and output filename. Keep the resource and formatting settings if they work well on your machine.

A seasonal association example, which deliberately allows common nouns as well as names:

```bat
python3 topic_wordlist.py "four seasons" --scope "Weather, nature, holidays, characters, activities, foods" --model qwen3.5:9b-q4_K_M --think off --categories 48 --rounds 3 --per-category 100 --batch-size 20 --context 8192 --max-tokens 2048 --max-words 3 --max-length 24 --styles words,compact --case lower --thorough -o seasons_qwen.txt
```

The manual-category pattern generalizes too: teams can be replaced by manufacturers, authors, countries, franchises, genres, or other explicit parent groups.

After the broad sports, cars, and automatic NFL runs have produced their text files, this command combines their exported lines and removes exact duplicates:

```bat
python3 -c "from pathlib import Path; files=['sports_qwen.txt','cars_qwen.txt','nfl_players_qwen.txt']; values=sorted({line.strip() for name in files for line in Path(name).read_text(encoding='utf-8-sig').splitlines() if line.strip()}); Path('combined_wordlist.txt').write_text('\n'.join(values)+'\n',encoding='utf-8'); print(len(values),'unique lines written')"
```

Change the input filenames if you used different examples. All named inputs must exist. This writes or replaces `combined_wordlist.txt`; select another destination name if you want to preserve an existing combined file.

The combined text file has no generation checkpoint. Continue expanding the original runs and then combine their outputs again. This merge removes exact duplicate lines, not equivalent spellings or formatting variants.

## 16. References and validation

This README describes the implemented v1.5.0 interface. Commands were checked against the script's argument parser and configuration rules. Live Qwen output quality and complete roster/catalog coverage were not tested as part of this documentation.

Official references checked October 22, 2026:

- [Qwen3.5 9B Q4_K_M package](https://ollama.com/library/qwen3.5:9b-q4_K_M)
- [Ollama quickstart](https://docs.ollama.com/quickstart)
- [Ollama FAQ: GPU loading, updates, and runtime settings](https://docs.ollama.com/faq)
- [NFL team directory](https://www.nfl.com/teams/)
- [2026 NFL rulebook: team name table](https://operations.nfl.com/rules-officiating/2026-nfl-rulebook)
- [Pro Football Hall of Fame directory](https://www.profootballhof.com/players)

The topic examples, scope strings, and category file are inputs to the existing generic workflow. No sports-specific implementation changes are required.
