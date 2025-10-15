# BIV Documentation

This directory contains detailed documentation for the BIV (Biologically Implausible Values) project:

## TDD Protocol

[`docs/tdd_protocol.md`](tdd_protocol.md) - Core Test-Driven Development protocol and best practices for coding agents implementing TDD in Python.

## Project Setup

[`docs/project_setup.md`](project_setup.md) - Comprehensive setup guide for Python projects using uv workflow, including:
- Environment initialization
- Dependency management
- Tool configuration (pytest, ruff, mypy, coverage)
- Pre-commit hooks
- CI/CD integration
- Additional tools (benchmarking, async testing, property-based testing)

## Version Control Workflows

[`docs/version_control_workflows.md`](version_control_workflows.md) - Git workflows, branching strategies, and development processes integrated with TDD:
- Feature and version branches
- Commit conventions
- Pull request process
- Conflict resolution
- Integration with TDD cycles
- Advanced workflows (Git Flow, GitHub Flow)

## CDC Growth Charts Reference

[`docs/cdc/`](cdc/) - Official documentation and data files from the U.S. Centers for Disease Control and Prevention (CDC):

- **[Extended BMI-for-Age Growth Charts](cdc/background-cdc-extended-bmi-for-age-growth-charts.md)** - Background on 2022 CDC Extended BMI charts for monitoring severe obesity in children
- **[Data Files Documentation](cdc/data-file-cdc-extended-bmi-for-age-growth-charts.md)** - Technical details on extended BMI calculation methods, LMS parameters, and percentile formulas
- **[Modified Z-Scores](cdc/modified-z-scores.md)** - Detailed explanation of modified z-score methodology for identifying biologically implausible values (BIVs)
- **[CDC SAS Program Documentation](cdc/sas-program-for-cdc-growth-charts.md)** - Technical guide for CDC's SAS programs used to calculate pediatric growth percentiles and z-scores
- **[WHO SAS Program Documentation](cdc/sas-program-for-who-growth-charts.md)** - Technical guide for WHO's SAS programs for infants and children under 2 years

This documentation provides authoritative references for the statistical methods, data sources, and calculation algorithms used in pediatric growth assessment.
