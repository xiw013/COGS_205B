# How to run the tests for the BayesFactor Object

## Running with Docker

Navigate to the project directory:

Build the image:

```bash
docker build -t bayes-factor .
```

Run the container:

```bash
docker run -dit \
       --name bayes_factor \
       --publish 2222:22 \
       -v $(pwd):/workspace \
       bayes-factor
```

Inside the container:

```bash
cd /workspace/repo
python3 -m unittest tests/test_bayes_factor.py
```