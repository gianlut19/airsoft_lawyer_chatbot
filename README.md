
# airsoft lawyer chatbot

This project implements a chatbot designed to assist users in understanding the PCR (Pattuglia corto raggio) regulations. The chatbot utilizes language models (Mistral) and retrieval-based question answering techniques to provide informative responses to user queries.

## Setup


### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/gianlut19/airsoft_lawyer_chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your_repository
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Obtain a Hugging Face token and insert it into the `HUGGING_TOKEN` variable within the code.

2. Ensure that the necessary PDF document containing the PCR regulations is available. The code expects the file to be located at `"app\PCR FIGT CRL - Campionato Regionale 2024-2025.pdf"`. Adjust the file path if necessary.

3. Run the application:

   ```bash
   streamlit run app/main.py
   ```

4. Interact with the chatbot by typing your queries into the input box and receiving responses.

## Functionality

The chatbot provides the following features:

- **Querying**: Users can input questions related to PCR regulations.
- **Response**: The chatbot generates responses based on the input query using a retrieval-based question answering approach.
- **History**: Previous interactions are displayed in the chat interface for reference.

## Components

### Language Models

- The chatbot leverages the Hugging Face Transformers library to interact with language models.
- It utilizes the Mistral-7B-Instruct-v0.2 model for generating responses.

### Vector Stores

- The chatbot uses sentence embeddings created with the Sentence Transformer library.
- It stores and retrieves information using the Chroma vector store.

### Persistence

- Data related to the PDF document and vector embeddings are persisted locally to optimize performance.

## Contributing

Contributions to the project are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the developers of the Hugging Face Transformers and Sentence Transformer libraries for their invaluable contributions to this project.