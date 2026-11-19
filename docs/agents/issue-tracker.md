# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `DCZJtAWhH/support-agent`. Use the `gh` CLI from this clone, or pass `--repo DCZJtAWhH/support-agent` when working elsewhere.

## Conventions

- Create an issue: `gh issue create --title "..." --body-file <path>`. Write multiline bodies to a UTF-8 file first.
- Read an issue: `gh issue view <number> --json number,title,body,labels,comments`.
- List issues: `gh issue list --state open --json number,title,body,labels,comments`, adding label and state filters as needed.
- Comment: `gh issue comment <number> --body-file <path>`.
- Apply or remove labels: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close: `gh issue close <number> --comment "..."`.

When a skill says "publish to the issue tracker", create a GitHub issue. When it says "fetch the relevant ticket", read the issue and its comments.

## Pull requests as a triage surface

**PRs as a request surface: no.**

If changed to `yes`, use `gh pr view`, `gh pr diff`, `gh pr comment`, `gh pr edit`, and `gh pr close` for the equivalent operations. GitHub issues and PRs share a number space; resolve ambiguous references with `gh pr view <number>`, falling back to `gh issue view <number>`.

## Wayfinding operations

- A map is one issue labelled `wayfinder:map`; its body holds Notes, Decisions-so-far, and Fog.
- Link child tickets as GitHub sub-issues. Obtain the child's database ID with `gh api repos/DCZJtAWhH/support-agent/issues/<child> --jq .id`, then link it with `gh api --method POST repos/DCZJtAWhH/support-agent/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>`. If unavailable, add the child to a task list in the map and put `Part of #<parent>` at the top of the child body.
- Child ticket types use `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.
- Record native dependencies with `gh api --method POST repos/DCZJtAWhH/support-agent/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`. Use the blocker's database ID, obtained as above. If unavailable, put `Blocked by: #<number>` at the top of the child body.
- The frontier is the first open child in map order with no open blockers and no assignee.
- Claim a ticket with `gh issue edit <number> --add-assignee @me`.
- Resolve by commenting with the answer, closing the ticket, and appending a gist and link to the map's Decisions-so-far.
