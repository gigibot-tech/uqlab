# Spec-Kit Setup for UQLab

## Overview
This document describes the Spec-Driven Development (SDD) setup for the UQLab ML Core library using GitHub's Spec-Kit framework.

## Installation

Spec-Kit v0.10.2 is installed and configured for this project.

```bash
# Verify installation
specify check

# Check for updates
specify self check
```

## Project Structure

```
uqlab/
├── .bob/                    # Bob AI agent integration
│   └── commands/            # Slash commands (/speckit.*)
├── .specify/                # Spec-Kit configuration
│   ├── memory/
│   │   └── constitution.md  # Project principles & standards
│   ├── specs/               # Feature specifications
│   │   └── 0001-fix-import-errors/
│   │       └── spec.md
│   ├── templates/           # SDD workflow templates
│   ├── scripts/             # Automation scripts
│   └── workflows/           # Workflow definitions
└── [existing code structure]
```

## Constitution Highlights

The UQLab constitution enforces:

### 1. **Modularity-First Architecture**
- Single Responsibility Principle
- Clear module boundaries
- Dependency injection over hardcoding
- Layered architecture (Data → Models → Training → Evaluation)

### 2. **Code Reuse Mandate** (NON-NEGOTIABLE)
Before creating new code, you MUST:
- Search existing codebase
- Check if utilities can be extended
- Verify no duplicates exist
- Document reuse decision

### 3. **DRY Principle**
Extract common patterns into:
- Shared utilities
- Base classes
- Factory functions
- Configuration objects

### 4. **Type Safety & Documentation**
- Type hints on all public APIs
- NumPy/Google style docstrings
- Usage examples in docs
- Runtime validation where critical

### 5. **Testing Standards**
- 80% minimum coverage
- Unit + integration tests
- Reusable fixtures
- Mock external dependencies

### 6. **Dependency Management**
- Minimize dependencies
- Pin versions
- Optional extras
- Python 3.11+ support

### 7. **Performance & Scalability**
- Profile before optimizing
- Batch processing support
- Memory-efficient generators
- GPU with CPU fallback

## Available Commands

### Core Workflow
```bash
/speckit.constitution  # 1. Define project principles (DONE)
/speckit.specify       # 2. Create feature specification
/speckit.plan          # 3. Create technical implementation plan
/speckit.tasks         # 4. Generate actionable task list
/speckit.implement     # 5. Execute implementation
```

### Optional Quality Gates
```bash
/speckit.clarify       # Resolve ambiguities (before /speckit.plan)
/speckit.checklist     # Quality validation (after /speckit.plan)
/speckit.analyze       # Coverage analysis (after /speckit.tasks)
```

### Utility Commands
```bash
/speckit.taskstoissues        # Convert tasks to GitHub issues
/speckit.agent-context.update # Update agent context
```

## Current Specifications

### Spec 0001: Fix Import Errors
**Status:** Specified (ready for planning)  
**Priority:** Critical  
**Location:** `.specify/specs/0001-fix-import-errors/spec.md`

**Scope:**
- Fix `ui_components/results/__init__.py` exports
- Fix `hypothesis_validation.py` imports
- Add comprehensive import tests
- Maintain backward compatibility

**Next Steps:**
1. Run `/speckit.plan` to create implementation plan
2. Run `/speckit.tasks` to generate task list
3. Run `/speckit.implement` to execute fixes

## Workflow Example

### Complete Feature Development Flow

```bash
# 1. Create specification
/speckit.specify
# Describe: "Add ResNet feature-space training mode"

# 2. Optional: Clarify ambiguities
/speckit.clarify

# 3. Create technical plan
/speckit.plan
# Specify: "Use PyTorch, freeze backbone layers, add config flag"

# 4. Optional: Generate quality checklist
/speckit.checklist

# 5. Generate tasks
/speckit.tasks

# 6. Optional: Validate coverage
/speckit.analyze

# 7. Implement
/speckit.implement
# Executes tasks systematically, pauses every 5 tasks for review
```

## Integration with Bob

Bob AI agent has access to all `/speckit.*` commands. Simply use them in chat:

```
You: /speckit.specify

Fix the import errors in ui_components that prevent clean package imports

Bob: [Creates detailed specification following template]
```

## Best Practices

### 1. Always Start with Constitution
Review constitution before starting any work to ensure compliance.

### 2. Spec Before Code
Never write code without a specification. Even small fixes benefit from structured thinking.

### 3. Use Reuse Checklist
Before creating new code:
- [ ] Searched for similar functionality
- [ ] Checked shared utilities
- [ ] Reviewed related modules
- [ ] Documented why reuse wasn't possible

### 4. Incremental Implementation
Use `/speckit.implement` which pauses every 5 tasks for review. This prevents runaway changes.

### 5. Quality Gates
For critical features, use all optional commands:
- `/speckit.clarify` - De-risk ambiguities
- `/speckit.checklist` - Validate requirements
- `/speckit.analyze` - Check coverage

## Troubleshooting

### Command Not Found
```bash
# Verify Spec-Kit is installed
specify check

# Reinstall if needed
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git@v0.10.2
```

### Bob Can't See Commands
```bash
# Verify .bob/commands/ exists
ls -la .bob/commands/

# Should show speckit.*.md files
```

### Spec-Kit Updates
```bash
# Check for updates
specify self check

# Upgrade to latest
specify self upgrade
```

## Resources

- **Spec-Kit Repository:** https://github.com/github/spec-kit
- **Documentation:** https://github.github.io/spec-kit/
- **Constitution:** `.specify/memory/constitution.md`
- **Specifications:** `.specify/specs/`

## Changelog

- **2026-06-15**: Initial Spec-Kit setup
  - Installed v0.10.2
  - Created constitution with modularity & reuse focus
  - Created Spec 0001 (Fix Import Errors)
  - Configured Bob integration