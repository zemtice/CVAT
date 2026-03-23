# Instructions to Setup Python Development Environment
Author: CYang, Date: 2026/02/24

This is an instruction guide to help you set up your Python development environment.
I will use **Visual Studio Code** and **Miniconda** as examples.

## Visual Studio Code (VSCode)
I recommend **VSCode** as your Python script editor. **Jupyter Lab** or **Jupyter Notebook** is also recommended. But you can use whatever IDE you favor.
- Step 1: Download and install VSCode: [https://code.visualstudio.com/](https://code.visualstudio.com/)
- Step 2: After you install VSCode, we can install a plugin for Python development. Select the **Extensions** tab in the left sidebar, search for Python and install the **Python** plugin:
![alt text](python_plugin.png)

## Miniconda
You might have heard about **Anaconda** before. **Miniconda** is a lightweight version of Anaconda, but it is still a handy tool to help us manage Python environments and packages.
Check this page to install **Miniconda**: https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions

## Create Python Environment
After installing **Miniconda**, you can use the conda command to create a virtual Python environment. All virtual environments you create are independent, so we can manage each project's dependencies easily with conda.

Command to create a virtual environment:
```shell
conda create -n <your_env_name> python=<specific_python_version>
```
For example, if I want to create an environment named "CV" with Python version "3.12", it will be: ```conda create -n CV python=3.12```.

To activate the environment you created:
```shell
conda activate <your_env_name>
```

And this is the deactivate command:
```shell
conda deactivate
```

While the environment is activated, you can install packages in it using the **pip** command:
```shell
pip install numpy opencv-contrib-python matplotlib
```
The command above will install numpy, opencv, and matplotlib. These packages are essential for this semester.

### Troubleshooting
**conda command not found**: You can open the Anaconda Prompt and run the command: ```conda init```, then open a new terminal — the conda command should be available.

> **Note for Mac/Linux users**: You may need to run ```conda init zsh``` or ```conda init bash``` depending on your shell.

## Run Python Script in VSCode
After you have created the environment, you can run Python scripts in VSCode. But first, let's create an example Python script.
- Step 1: Reopen VSCode, select "Open Folder" in the welcome tab, and choose the folder where you want to save your first Python script.
- Step 2: Create a text file in the folder and name it with a `.py` suffix.
- Step 3: In this file, add ```print("Hello world")```, and save by pressing Ctrl+S / Command+S.

After these steps, you may notice a **"Select Python Interpreter"** prompt in the bottom right corner of VSCode. Click on it and choose the environment you created in the previous steps from the popup list. After selecting, you can run your script by clicking the ▶️ button in the top right corner.

## Course Resources on GitHub Repository
Besides creating Python scripts on your computer, there is another way to get started. When you open VSCode for the first time, you will see another option in the welcome tab called "Clone Git Repository". This means you can "download" someone's project that is publicly available on https://www.github.com. In this course, the TA will put all resources and the scripts generated in class on a GitHub repository. You can clone the repository with VSCode:
- Step 1: Click "Clone Git Repository" on the welcome tab.
- Step 2: https://github.com/zemtice/CVAT.git is the URL of this course repository. Enter this URL in the popup input box.
- Step 3: Select the location on your computer where you want to save the repository.

After these steps, the repository will automatically be downloaded to your computer. This repository will be updated according to the progress of the course. If you want to sync the content, you can **Pull (rebase)** from the GitHub repository to your local repository. We will talk about more Git usage in the future.
