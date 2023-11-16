# COMP4989_Proj_Open_Data

## Table of Contents

-   [Installation](#installation)
    -   [Virtual environment setup](#virtual-environment-setup)
    -   [Create a virtual environment](#create-a-virtual-environment)
    -   [Activate the virtual environment](#activate-the-virtual-environment)
    -   [Installing dependencies](#installing-dependencies)

## Installation

### Virtual environment setup

To manage the project dependencies and isolate them from your system-wide packages, it's recommended to use a virtual environment. If you haven't installed `virtualenv`, you can do so using:

```bash
pip install virtualenv
```

### Create a virtual environment:

On Unix or MacOS

```bash
python3 -m venv venv
```

On Windows

```bash
python -m venv venv
```

### Activate the virtual environment:

MacOS/Linux

```sh
source venv/bin/activate
```

Windows (cmd)

```sh
venv\Scripts\activate.bat
```

Windows (PowerShell)

```sh
venv\Scripts\Activate.ps1
```

### Installing Dependencies

Once the virtual environment is activated, install the project dependencies from the requirements.txt file:

```sh
   pip install -r requirements.txt
```
