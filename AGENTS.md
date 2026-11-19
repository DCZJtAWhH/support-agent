## File encoding

Read text files with UTF-8 encoding explicitly. In PowerShell, use `Get-Content -LiteralPath <path> -Encoding UTF8`.

## 中文注释

`src/` 目录及其子目录中的每个函数和方法（包括构造方法、私有方法和异步方法）都必须在定义处写中文注释或文档字符串，说明其用途；参数、返回值或副作用含义不直观时，一并说明。

完成代码修改前，逐一检查所修改文件中的函数和方法，确保每个都有与当前实现一致的中文注释。

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues for `DCZJtAWhH/support-agent`. Before reading or publishing tickets, read `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. Before triaging issues, read `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `GLOSSARY.md` and `docs/adr/`. Before exploring the codebase, read `docs/agents/domain.md`.
