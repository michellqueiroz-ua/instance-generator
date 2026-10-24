# Project File Format Reference

This document describes the JSON schema for project files located in
`data/users/<user>/projects/<project_slug>.json`.
Use this as the specification when generating a new project file for the research scheduler.

---

## Top-level fields

```json
{
  "id": "SHORT-PREFIX",
  "project_name": "Full Human-Readable Project Name",
  "context": "...",
  "priority_level": 3,
  "tasks": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Short ALL-CAPS prefix used as the namespace for all task IDs in this project. E.g. `"FUNDING"`, `"ISA-HODBRP"`, `"SSS-STOCH"`. Must be unique across projects. |
| `project_name` | string | Full display name shown in the UI. |
| `context` | string | 2–5 sentence description of the project: what it is, why it matters, key constraints (deadlines, stakeholders, scope limits). This text is shown in the project card and can be passed to an LLM as background. |
| `priority_level` | integer | 1 (lowest) – 5 (highest). Controls scheduling order: higher-priority projects fill earlier slots in the weekly calendar. Default: `3`. |
| `tasks` | array | Ordered list of task objects (see below). |

---

## Task object fields

```json
{
  "id": "PREFIX-T01",
  "name": "Short action-oriented task name",
  "order": 1,
  "estimated_duration": 1.5,
  "tags": ["tag1", "tag2"],
  "cognitive_load": "high",
  "context": "...",
  "prompt_file": "prompts/my_prompt.md",
  "status": "not_started",
  "depends_on": ["PREFIX-T00"],
  "deadline": "2026-06-03",
  "task_type": "standard"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✅ | Unique task identifier. Convention: `<PROJECT-ID>-<code>`. The code can be sequential (`T01`, `T02`) or descriptive (`V1`, `A-D1`). Must be globally unique across all projects. |
| `name` | string | ✅ | Short task name shown in the UI. Use an action verb: "Draft introduction", "Run SAA experiment", "Review R2 comments". Keep under ~60 chars. |
| `order` | integer | ✅ | Sort position within the project (1-based). Controls display order and scheduling order. |
| `estimated_duration` | number | ✅ | Estimated work time in **hours** (decimals allowed). Use `0.5` for 30 min, `1.5` for 90 min. Tasks > 3 h are automatically split into ≤ 2 h calendar chunks. |
| `tags` | array of strings | ✅ | Free-form labels for filtering/grouping. Common values: `"writing"`, `"code"`, `"reading"`, `"experiments"`, `"admin"`, `"decision"`, `"communication"`, `"review-response"`, `"setup"`. |
| `cognitive_load` | string | ✅ | Determines **time-of-day scheduling band**. One of: `"high"` → morning (09:00–12:00), `"medium"` → midday (12:00–14:00), `"low"` → afternoon (14:00–17:00). Use `"high"` for deep work (writing, coding), `"medium"` for analysis/reading, `"low"` for admin/decisions. |
| `context` | string | ✅ | 2–6 sentence description of what this task specifically requires, what inputs are needed, what the expected output is, and any stopping criteria. This is the most important field for an LLM — it should be self-contained enough to act on without reading the rest of the project. |
| `prompt_file` | string or null | ✅ | Relative path (from project root) to a `.md` prompt file that provides a full LLM prompt for this task. Set to `null` if no prompt file exists. E.g. `"prompts/fund_v1_gap_analysis.md"`. |
| `status` | string | ✅ | Current task status. One of: `"not_started"`, `"in_progress"`, `"done"`, `"blocked"`. New tasks should always start as `"not_started"`. |
| `depends_on` | array of strings | ✅ | List of task IDs that must be `"done"` before this task appears in the weekly calendar. Use `[]` for tasks with no prerequisites. E.g. `["FUND-0", "FUND-V0"]`. Can reference tasks from other projects. |
| `deadline` | string or null | ✅ | Hard deadline in `"YYYY-MM-DD"` format, or `null` if none. Shown as a colored badge in the UI (red if overdue, yellow if ≤ 7 days away). |
| `task_type` | string | optional | `"standard"` (default) or `"checkpoint"`. Use `"checkpoint"` for recurrent review/check tasks (e.g. "collect HPC results") that may loop back — they get a ⟳ icon and are never split across chunks. Omit the field to default to `"standard"`. |

### Runtime fields (written by the app — do not set manually)

These fields are added automatically when the user interacts with the UI or timer:

| Field | Written by |
|---|---|
| `time_spent_seconds` | Timer start/stop |
| `timer_started_at` | Timer start (ISO 8601 UTC string, or `null`) |
| `what_i_did` | Modal note |
| `what_is_blocking` | Modal note |
| `next_step` | Modal note |
| `notes` | Modal scratch notes |

---

## Conventions

- **Task ID uniqueness**: IDs must be unique across ALL project files, not just within one project, because `depends_on` references are resolved globally.
- **Ordering**: `order` values should be sequential integers starting at 1. Gaps are allowed but not recommended.
- **Task granularity**: aim for tasks of 0.5–3 hours. If a task would naturally take longer, split it into sequential sub-tasks (e.g. `V1`, `V2`, `V3` for a multi-session writing arc).
- **Context quality**: the `context` field for each task should answer: *what exactly do I do*, *what do I use as input*, *what does done look like*, and *where does the output go*. Avoid vague verbs like "work on" or "think about".
- **depends_on discipline**: only add a hard dependency when the task is genuinely unblockable without the prior one. Loose thematic sequence should be expressed via `order`, not `depends_on`.

---

## Minimal valid example

```json
{
  "id": "MYPROJ",
  "project_name": "My New Project",
  "context": "One paragraph describing what this project is, why it matters, and any key constraints or deadlines.",
  "priority_level": 3,
  "tasks": [
    {
      "id": "MYPROJ-T01",
      "name": "Define scope and deliverables",
      "order": 1,
      "estimated_duration": 1.0,
      "tags": ["admin", "decision"],
      "cognitive_load": "medium",
      "context": "Write a one-page scope doc: what is in/out of scope, who the audience is, and what the final deliverable looks like. Output goes to docs/scope.md.",
      "prompt_file": null,
      "status": "not_started",
      "depends_on": [],
      "deadline": null
    },
    {
      "id": "MYPROJ-T02",
      "name": "Draft first section",
      "order": 2,
      "estimated_duration": 2.0,
      "tags": ["writing"],
      "cognitive_load": "high",
      "context": "Based on the scope doc from T01, write the first section. Target: 500–800 words, complete argument, no placeholders. Output to docs/draft_v1.md.",
      "prompt_file": "prompts/myproj_t02_draft.md",
      "status": "not_started",
      "depends_on": ["MYPROJ-T01"],
      "deadline": "2026-07-01"
    }
  ]
}
```

---

## LLM instructions for generating a new project file

When asked to generate a project file, follow these steps:

1. **Derive the `id`** from the project name: use a short ALL-CAPS abbreviation (2–10 chars, hyphens allowed, no spaces).
2. **Write a dense `context`** for the project: include the goal, the main output, key constraints (deadlines, collaborators, word counts, tool requirements), and any scope limits.
3. **Break the project into tasks** of 0.5–3 hours each. If unsure, err on the side of smaller tasks.
4. **Set `cognitive_load` deliberately**: reserve `"high"` for writing and coding; use `"medium"` for analysis and reading; use `"low"` for admin, decisions, and communication.
5. **Wire `depends_on`** only for hard sequential dependencies. The first task in every project usually has `"depends_on": []`.
6. **Set all `status` to `"not_started"`** and `"timer_started_at"` / time fields should be omitted (the app adds them on first use).
7. **Save to** `data/users/<user>/projects/<project_slug>.json` where `project_slug` is a lowercase, hyphen-separated version of the project ID.
