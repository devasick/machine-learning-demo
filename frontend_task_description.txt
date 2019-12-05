Task description
----------------
The file task_data.csv contains an example data set that has been artificially
generated. The set consists of 400 samples where for each sample there are 10
different sensor readings available. The samples have been divided into two
classes where the class label is either 1 or -1.

You will also find a boilerplate for a backend application that consists of
following files: main.py, templates/base.html, templates/index.html and 
requirements.txt
This application reads the provided task_data.csv and converts it into a
JSON object when /data is requested.

The backend application is implemented in python using:
 - Flask web app framework https://flask.palletsprojects.com/en/1.1.x/
 - Jinja2 templating engine https://jinja.palletsprojects.com/en/2.10.x/

Your task is to create intuitive page(s) that contain informative visualizations 
of the provided data. Use HTML, JS and CSS to read the JSON and display it in an
organized way. The aim of the page(s) and visualizations should be to explore 
and understand the data for a statistical analysis.
- You may use a charting library of your choice, preferably Highcharts.
- You may use a user interface framework if you think it is necessary.
- You may change the backend python application code
- You should *not* change the task_data.csv file

Additionally, please include a simple description document where you describe:
    * your choices of libraries and frameworks (if any)
    * your choice of the design language and its influences
    * strengths of your design: What are the key points that separates your 
            design from a standard approach?
    * weaknesses of your design: What tradeoffs you had to make? What would you
            like to change for a better result?


Instructions
------------
Using Python 3.6.7
1. Create a virtualenv "venv" with 'python3 -m venv venv'. If you do not have
    virtualenv python package installed, follow the installation manual:
    https://virtualenv.pypa.io/en/stable/installation/
2. Activate it with '. venv/bin/activate'
3. Install the requirements with 'pip install -r requirements.txt'
4. Run the application with 'python3 -m main'


Technical requirements
----------------------
Please submit *one* zip file that contains the following:
    * The full flask application
    * The description document
It should be possible to run your solution using the same instructions above.
