Run the tools on the provided samples:

```
python run_samples.py "python factory/main.py" "python belts/main.py"
```

Execute the test suite (expects `pytest` on PATH):

```
FACTORY_CMD="python factory/main.py" BELTS_CMD="python belts/main.py" pytest -q
```
