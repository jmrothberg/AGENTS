---
name: research
description: Run a structured multi-source research pass and write a cited brief.
triggers: research, deep-dive, investigate, brief, summarize the web on
---

# Research Skill

When this skill is invoked, follow this process exactly. Do not skip steps.

## 1. Restate the question

In one sentence, write back what the user actually wants to know. If the
question is ambiguous, ask one clarifying question before continuing.

## 2. Plan the sources

List 3–5 concrete sources you will consult. Prefer this order:
1. Primary sources (official docs, papers, filings, press releases)
2. Reputable secondary sources (well-known publications, maintained wikis)
3. Community discussion (forums, issue trackers) — only to surface edge cases

## 3. Gather

For each source, use the most appropriate tool:
- Structured API or JSON endpoint → `fetch_url`
- Web pages that need JavaScript or login → `browser_goto` + `browser_read`
- Local files → `read_file`
- Your own memory → `recall_memory`

Write quoted snippets + the URL into a scratch note so you can cite them
later. If a source contradicts another, keep both.

## 4. Synthesize

Write a brief with this structure:

    # <Topic>

    **Bottom line:** <one-sentence answer>

    ## Key findings
    - finding (source)
    - finding (source)

    ## Open questions
    - question you could not resolve

    ## Sources
    1. <title> — <url>
    2. <title> — <url>

Every factual claim must link to at least one source in the list. If you
could not verify a claim, say so explicitly instead of guessing.

## 5. Save

- Write the brief to `workspace/research/<slug>.md` using `write_file`.
- Save one memorable fact per finding with `add_task` (priority=low,
  status=done) so future runs can `recall_memory` on it.
- Return the brief to the user along with the file path.
