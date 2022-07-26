# Project description
In this project we implement different information retrieval and analysis application on Shahname.
# Project structure
This repository is split into two parts: CLI codes for different search methods and a unified GUI.<br>
Both parts include 5 methods: boolean, TF-IDF, Transformer based, fasttext based and elasticsearch. <br>
# Program execution
In the CLI case, the first four methods mentioned above are mainly implemented in one python file (appropriately named for each method), and you only need to run this file to use the method. For example, to use the boolean method, you can use the following command:
```
python -i boolean_retrieval.py
```
This initializes the values for this method and opens a python shell, and using
```
search(query,k)
```
in this command returns the expected results. <br>
For elasticsearch, before running the related pyrhon file, we first need to launch elasticsearch on port 9200 (the default port for elasticsearch) and the rest of the process is the same as the other methods. Elasticsearch can be downloaded from [elastic]([https://elastic.co](https://www.elastic.co/downloads/elasticsearch)) and launched by running the executable file in the bin folder.<br>
For the GUI, we again first need to launch elasticsearch. Afterwards, we need to execute the commmand 
```
python manage.py runserver
```
in "GUI/shahname" to launch the web-based user interface. This interface allows the user to search the verses of Shahname using the aforementioned methods. 

