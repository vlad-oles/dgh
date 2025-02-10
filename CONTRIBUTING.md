# Contributing to dgh
## 1. How to contribute

### 1.1. Fork & clone the repository
1. Fork this repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/dgh.git
   cd dgh
   ```
3. Add the upstream repository:
    ```bash
    git remote add upstream https://github.com/vlad-oles/dgh.git
    ```
4. Make sure your repo is up-to-date:
    ```bash
    git fetch upstream
    ```

### 1.2. Setup the environment
1. Install the package in editable mode with dependencies:
    ```bash
   pip install -e .
    ```
2. Create a new branch for your contributions:
    ```bash
    git checkout -b feature-branch-name
   ```
   Use meaningful names such as optimize-matrix-multiplication or fix-issue-42.

### 1.3. Modify the code
1. Make your changes.
2. Make sure that they don't break the test suite:
    ```bash
   pytest
    ```

### 1.4 Submit a pull request
1. Push your changes.
    ```bash
   git push origin feature-branch-name
    ```
2. Open a pull request (PR) on GitHub, describing:
   * the problem being solved;
   * the approach used;
   * any potential issues or trade-offs.

## 2. Review & merging process
* Your PR will be reviewed and may require changes.
* Once approved, it will be merged into the main branch.
* If your contribution is substantial, your name will be added to CONTRIBUTORS.md.

## 3. Contributor recognition

All accepted contributors will be credited in:
* The GitHub contributors list.
* The CONTRIBUTORS.md file (for major contributions).
* The package metadata (for long-term maintainers).

Thank you for contributing!