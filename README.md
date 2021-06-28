# MDS Capstone Report: Diagnosing Lipohypertrophy
Capstone project in collaboration with the Gerontology and Diabetes Research Laboratory (GDRL). Using Machine Learning Approaches to Diagnosing Lipohypertrophy at the bedside.
  - contributors: Ela Bandari, Lara Habashy, Javairia Raza, Peter Yang

## About
This repository hosts all the files and scripts needed to run, generate, and publish the final capstone project report written by Ela Bandari, Lara Habashy, Javairia Raza, and Peter Yang. The report is published on GitHub pages via Jupyter Book. The most recent deployment of this report took place on June 28th, 2021 at 11:00AM PST. 

## Usage
To re-generate this report, follow these steps:
1. First, you will need to install Jupyter Book via pip. Refer to [this](https://jupyterbook.org/intro.html) manual to get started. <br> Ensure Anaconda Python is installed.
2. Clone this repository by running the following commands from your terminal line:
<br>`git clone https://github.com/larahabashy/capstone-report.git`
<br>`cd capstone-report`
3. Install our environment by running:
<br>`conda env create -f environment.yml`
<br>`conda activate report_env`
4. Compile the report component by running the following command from the root directory:
<br>`jupyter-book build ./capstone-report` <br> This allows you to view an html rendered version of the book on your browser of choice
5. Publish the report on GitHub pages by running the following two commands:
<br> `pip install ghp-import`
<br> `ghp-import -n -p -f _build/html`

For more information, visit the Jupyter Book tutorial page [here](https://jupyterbook.org/start/your-first-book.html).

## License and Code of Conduct
The project uses MIT license. It can be found
[here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/LICENSE).

The code of conduct can be found 
[here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/CODE_OF_CONDUCT.md).


