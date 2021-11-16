# RL Trading

## Prerequisites

1. pipenv
1. (Optional) jupyter
1. (Optional) VSCode

## Installation

Run `pipenv install` to install all the required dependencies

## Running

Run `main.py` using `python src/main.py`. Keep in mind that you have to be in this project's root while executing the command.

## Debugging in VSCode

Paste the following to you `launch.json` file:

```
...
"configurations": [
  {
    "name": "Python: Main",
    "type": "python",
    "request": "launch",
    "program": "src/main.py",
    "console": "integratedTerminal"
  }
]
...
```
