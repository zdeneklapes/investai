# AI-INVESTING: Bachelor Thesis

## README

---

### AUTHORS

- **Student**: Zdeněk Lapeš <lapes.zdenek@gmail.com>
- **Superviser**: Milan Češka <ceskam@fit.vut.cz>

### BUGS

### DEPENDENCIES

- swig
- ta-lib

### CONTRIBUTING

Please be patent what you are doing and why. If you are not sure, ask.
The `main` and `dev` branches are considered as protected. You can not
push directly to them. You have to create a pull request and wait for
review. If you are not sure, ask.

#### Before pull request

```bash
# Pre-commit
pre-commit install              # Run all before commit)
pre-commit install -t pre-push  # Run all before push)
pre-commit run --all-files      # Run all checks manually

# Run tests
pytest
```

#### INSTALLATION

```shell
# venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### [TODOS](./TODOS.md)

### [MATERIALS](./MATERIALS.md)
