## Coding Discipline (agents that write code)

Adapted in part from Ponytail by Dietrich Gebert (MIT), <https://github.com/DietrichGebert/ponytail>.

**Before writing**
- List every place the change must reach: callers, tests, fixtures, config, exports, docs. Name what it could break for users. Then write.
- Take the first of these that works: the code doesn't need to exist; it already exists in the codebase; the standard library or platform has it; a dependency already installed has it; write it yourself, as little as does the job. Adding a dependency needs the user's OK.
- For a bug fix, find every caller first, then fix the root cause once, in the shared code.

**Writing**
- Once you know everything a change must touch, the shortest diff wins. A one-liner that needs decoding is not short.
- No code "for later": no unused options, config knobs, or abstractions with one user. Exception: a shared interface landed first on purpose so two agents can build against it.
- Code you move or merge keeps its validation and error handling.
- Comment only the why that the code can't show.

**Never cut.** No rule above overrides these:
- validation where untrusted input enters (users, network, files, other agents)
- handling that prevents data loss
- security checks and fail-closed paths
- accessibility
- anything the user asked for

**Tests**
- Logic with a branch, loop, parser, money, security, or a new script leaves at least one test. Trivial changes (a rename, copy text, a config value) need none.
- Security and fail-closed paths get one test per decision case, not one test in total.

_This section is yours to edit. It is plain markdown in this file; change or remove any rule that doesn't fit how you work._

