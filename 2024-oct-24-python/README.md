# GPU programming using Python – A practical intro webinar

<https://enccs.se/events/gpu-programming-using-python-a-practical-intro/>

## About the webinar

In the past decade, Graphics Processing Units (GPUs) have ignited the dynamic evolution of data science. But GPUs can do a lot more than machine learning – these powerful devices can accelerate and massively parallelise any general-purpose computational load in domains involving big data and heavy number crunching. You can use the GPU in your personal computer, or scale up your application to run on a supercomputer. How can you get started?

In this webinar, we focus on GPU-accelerated computing with Python, one of the most popular programming languages for science, engineering, data analytics, and deep learning applications. Starting from familiar Python libraries such as Numpy and Pandas, we will guide you step-by-step into the world of GPU programming. Discover how to harness the power of GPU accelerators using libraries such as CuPy, cuDF, PyCUDA, Jax, and Numba, with a focus on their unique features and capabilities for high-performance computing.

## Who is the webinar for?

The GPU programming using Python webinar is for data scientists, software developers and researchers who want to start using GPUs to accelerate their computational workflows.

## Key takeaways

After attending this seminar, you will be able to:

- get started with GPUs in Python using high-level libraries
- familiarise yourself with the multiple vendors that compete with different software stacks, toolkits, and frameworks
- make informed decisions about your GPU workflows

## Installation

List of package can be installed using [pixi], [conda] or [pip]

[pixi]: https://pixi.sh
[conda]: https://conda-forge.org/download/
[pip]: https://pip.pypa.io/en/stable/installation/

### [pixi]

```
pixi run jupyter lab
```

### [conda]

```
conda env create -f environment.yaml
conda run -n gpu-python-webinar jupyter lab
```

### [pip]

Create a virtual environment, activate it and

```
pip install -r requirements.txt
jupyter lab
```

> [!WARNING]
> You may need to ensure that you have the CUDA SDK installed in the system.
