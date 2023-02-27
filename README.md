# OpenAI API Quickstart - Python example app

This is a POC on building a Product Information Kiosk  with OpenAI ChatGPT integration to answer questions about a product. It is based on the OpenAI API [quickstart tutorial](https://beta.openai.com/docs/quickstart). 
The application uses two OpenAI APIs: 
1. Embeddings API - to provide context to ChatGPT using document embeddings retrieval and 
2. Completions API - to answer the questions about the product. 

Sources - https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
https://github.com/openai/openai-quickstart-python

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Clone this repository

3. Navigate into the project directory

   ```bash
   $ cd openai-quickstart-python
   ```

4. Create a new virtual environment

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file

   ```bash
   $ cp .env.example .env
   ```

7. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file

8. Install tiktoken
      In your terminal, install tiktoken with pip: pip install tiktoken

9. Run the app

   ```bash
   $ flask run
   ```

You should now be able to access the app at [http://localhost:5000](http://localhost:5000)! For the full context behind this example app, check out the [tutorial](https://beta.openai.com/docs/quickstart).
