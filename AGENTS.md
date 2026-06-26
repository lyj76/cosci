# cosci project guidance

## Project scope

This repository is the `cosci` project located at:

`/mnt/e/llm/cosci`

Use this file as the project-level operating guide for Codex.

## General behavior

* Prefer small, testable changes.
* Do not make broad rewrites unless the task clearly requires them.
* Before editing code, inspect the relevant local files.
* After editing code, run the smallest relevant verification command when possible.
* Explain what changed, why it changed, and how it was verified.
* Do not store secrets, tokens, passwords, private keys, or raw credentials in any knowledge base.

## MCP usage policy

Use MCP tools deliberately, not excessively.

### Project memory: `kb_cosci`

Use `kb_cosci.qdrant-find` before non-trivial work involving:

* debugging
* architecture changes
* dependency changes
* environment changes
* MCP configuration
* recurring project-specific problems
* repository-specific decisions

Use `kb_cosci.qdrant-store` after solving a real issue or making a durable decision.

Store only verified, durable project knowledge.

Good things to store:

* final bug root cause
* working commands
* stable configuration
* architecture decisions
* project-specific file layout
* repository conventions
* environment notes that are likely to matter again

Do not store:

* guesses
* temporary attempts
* large raw logs
* unverified internet claims
* private secrets
* credentials
* one-off conversation context

When storing memory, use this format:

`[date=YYYY-MM-DD][project=cosci][type=bugfix|command|architecture|config|decision][confidence=high|medium][status=current|deprecated|experimental]`

Then include:

* short summary
* evidence, exact error, command, path, or context
* final resolution or lesson
* future retrieval keywords when useful

Important note:

If `/mcp` shows `kb_cosci` as `Tools: (none)`, do not assume the knowledge base is broken. Verify by directly calling `kb_cosci.qdrant-store` or `kb_cosci.qdrant-find`.

### Documentation: `context7`

Use `context7` when the task depends on current or precise documentation for:

* libraries
* frameworks
* APIs
* package behavior
* configuration syntax
* version-sensitive usage

Prefer `context7` over memory when API behavior may have changed.

Typical workflow:

1. Resolve the library ID with `context7.resolve-library-id`.
2. Query the docs with `context7.query-docs`.
3. Apply the result to the local code or configuration.

### GitHub: `github`

Use `github` only when the task involves GitHub-hosted information or actions:

* repository search
* issues
* pull requests
* commits
* releases
* branches
* GitHub code search
* creating or updating repository files
* pull request review or comments

Do not use GitHub MCP for facts that can be answered from local files.

Be careful with write tools such as:

* `create_or_update_file`
* `delete_file`
* `create_pull_request`
* `merge_pull_request`
* `push_files`
* `issue_write`
* `pull_request_review_write`

Before using a destructive or publishing GitHub action, explain the intended action and make sure it matches the user request.

### Paper search: `paper-search`

Use `paper-search` for academic or biomedical literature tasks involving:

* arXiv
* PubMed
* PMC
* Europe PMC
* Crossref
* Semantic Scholar
* OpenAlex
* DOI
* PMID
* PMCID
* literature reviews
* paper metadata
* paper reading and comparison

Preferred tool choices:

* Use `search_papers` for broad academic search.
* Use `search_arxiv` for ML, AI, CS, math, and preprints.
* Use `search_pubmed` for biomedical literature.
* Use `search_pmc` or `search_europepmc` when full text or biomedical open-access content is needed.
* Use `search_semantic` or `search_openalex` for citation-oriented discovery.
* Use `get_crossref_paper_by_doi` when a DOI is known.
* Use `read_*_paper` only after selecting a specific paper.
* Use `download_with_fallback` only after selecting a specific paper and when full text is needed.

Avoid using `download_scihub`.

When summarizing papers, include available identifiers:

* title
* authors
* year
* venue
* DOI
* arXiv ID
* PMID
* PMCID

For useful long-term paper notes, store a concise summary in the appropriate knowledge base only if a paper knowledge base exists. If no paper knowledge base is configured, do not force paper notes into `kb_cosci` unless they are directly relevant to this project.

## Workflow for debugging

Before debugging:

1. Inspect the local error message and relevant files.
2. Search `kb_cosci.qdrant-find` for similar prior issues.
3. Use `context7` if the problem depends on current package or API behavior.
4. Use web or GitHub only when the issue depends on external, current, or repository-hosted information.

During debugging:

* Prefer minimal reproductions.
* Change one thing at a time when possible.
* Keep track of final working commands.
* Do not preserve every failed attempt as memory.

After debugging:

* Summarize the root cause.
* Summarize the final fix.
* Mention verification performed.
* Store one durable memory in `kb_cosci.qdrant-store` if the lesson is likely to help again.

## Workflow for MCP configuration

When troubleshooting MCP:

1. Check `~/.codex/config.toml`.
2. Prefer absolute command paths when PATH may differ inside Codex.
3. For `uvx` MCP servers, prefer `/root/.local/bin/uvx`.
4. For stdio MCP servers, remember that manually running the server may appear to hang because it is waiting for JSON-RPC input.
5. If a stdio MCP server prints non-JSON logs to stdout, use a wrapper script and redirect stderr to a log file.
6. Verify by direct tool invocation, not only by `/mcp` display.

For Qdrant MCP:

* Use environment variables, not old command-line flags.
* Use `QDRANT_LOCAL_PATH`.
* Use `COLLECTION_NAME`.
* Use `EMBEDDING_MODEL`.
* Use `QDRANT_SEARCH_LIMIT`.
* Verify with direct calls to `qdrant-store` and `qdrant-find`.

## Memory quality rules

Only store knowledge that is:

* verified
* durable
* concise
* useful for future retrieval

Every stored memory should have:

* date
* project or domain
* type
* confidence
* status
* exact command, path, error message, or source context when useful

Prefer this structure:

`[date=YYYY-MM-DD][project=cosci][type=config][confidence=high][status=current]`

Summary:

* One or two sentences.

Evidence:

* Exact command, error, file path, or context.

Resolution:

* Final verified conclusion.

Future retrieval keywords:

* Short keywords likely to be searched later.

## What not to do

* Do not store secrets.
* Do not store raw tokens.
* Do not store private keys.
* Do not store temporary guesses.
* Do not store long logs.
* Do not use paper-search for normal coding questions.
* Do not use GitHub write tools unless the user requested a GitHub action.
* Do not assume `/mcp Tools: (none)` means a server is unusable.
* Do not overuse MCP tools when local files and verified memory are enough.
